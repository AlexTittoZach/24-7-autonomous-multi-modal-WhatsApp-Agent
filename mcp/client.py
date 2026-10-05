import asyncio
import json
import logging
import os
import shlex
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("MCPClient")


class StdioMCPClient:
    """
    Universal Model Context Protocol (MCP) Client over stdio (pipes).
    
    1. Spawns an external MCP server (Python, Node.js, Go, or Rust) as a subprocess.
    2. Communicates via JSON-RPC 2.0 over stdin/stdout pipes.
    3. Handles the official Anthropic MCP 3-step initialization handshake.
    4. Auto-discovers available tools via 'tools/list'.
    5. Dynamically registers them into Hermes Agent's ToolRegistry.
    """

    def __init__(
        self,
        command: str | List[str],
        name: str = "mcp-server",
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        if isinstance(command, str):
            self.command = shlex.split(command)
        else:
            self.command = command

        self.name = name
        self.env = {**os.environ, **(env or {})}
        self.cwd = cwd

        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def start(self):
        """Spawns the MCP server child process and completes the protocol handshake."""
        logger.info(f"🚀 Spawning MCP server '{self.name}': {' '.join(self.command)}")
        
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )

        # Start continuous background readers for stdout and stderr
        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())

        # Perform the official 3-step Anthropic MCP Handshake
        await self._perform_handshake()
        logger.info(f"✅ MCP server '{self.name}' successfully connected and initialized!")

    async def _perform_handshake(self):
        """
        Step 1: Send 'initialize'
        Step 2: Await server capabilities
        Step 3: Send 'notifications/initialized'
        """
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "whatsapp-hermes-agent",
                "version": "1.0.0",
            },
        }

        # Step 1: Initialize
        server_info = await self.send_request("initialize", init_params)
        logger.info(f"🤝 Handshake complete with '{self.name}': {server_info.get('serverInfo', {})}")

        # Step 2: Confirm initialized
        await self.send_notification("notifications/initialized")
        self._initialized = True

    async def _read_stdout_loop(self):
        """Reads JSON-RPC responses line-by-line from the server's stdout pipe."""
        try:
            while self.process and self.process.stdout and not self.process.stdout.at_eof():
                line = await self.process.stdout.readline()
                if not line:
                    break

                text = line.decode("utf-8").strip()
                if not text:
                    continue

                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    # Ignore non-JSON log noise that some servers emit to stdout
                    logger.debug(f"[{self.name} non-JSON stdout]: {text}")
                    continue

                # Handle response to a pending request
                msg_id = message.get("id")
                if msg_id in self._pending_requests:
                    future = self._pending_requests.pop(msg_id)
                    if not future.done():
                        if "error" in message:
                            future.set_exception(RuntimeError(f"MCP Error: {message['error']}"))
                        else:
                            future.set_result(message.get("result", {}))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in {self.name} stdout reader: {e}", exc_info=True)

    async def _read_stderr_loop(self):
        """Continuously reads and logs stderr so the OS pipe buffer never deadlocks."""
        try:
            while self.process and self.process.stderr and not self.process.stderr.at_eof():
                line = await self.process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.warning(f"[{self.name} stderr]: {text}")
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
        """Sends a JSON-RPC 2.0 request and waits for the server's response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"MCP server '{self.name}' is not running.")

        self._request_id += 1
        req_id = self._request_id

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future

        message_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        self.process.stdin.write(message_bytes)
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"MCP request '{method}' to '{self.name}' timed out after {timeout}s.")

    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Sends a JSON-RPC 2.0 notification (no response expected)."""
        if not self.process or not self.process.stdin:
            return

        payload = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        message_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        self.process.stdin.write(message_bytes)
        await self.process.stdin.drain()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Queries the server for its list of available tools and parameter schemas."""
        result = await self.send_request("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool on the external MCP server and extracts the text output."""
        params = {
            "name": name,
            "arguments": arguments,
        }
        result = await self.send_request("tools/call", params, timeout=45.0)

        # Parse MCP content array: [{"type": "text", "text": "..."}]
        content_items = result.get("content", [])
        output_parts = []
        for item in content_items:
            if item.get("type") == "text":
                output_parts.append(item.get("text", ""))
            else:
                output_parts.append(json.dumps(item))

        return "\n".join(output_parts) if output_parts else "Success (No text output)"

    async def register_tools(self, registry: Any) -> List[str]:
        """
        Auto-discovers tools from the MCP server and registers them into Hermes's ToolRegistry.
        Returns the list of registered tool names.
        """
        tools = await self.list_tools()
        registered_names = []

        for tool in tools:
            tool_name = tool.get("name")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})

            # Closure that forwards LLM tool invocations to the MCP server
            async def make_executor(t_name=tool_name):
                async def executor(**kwargs):
                    return await self.call_tool(t_name, kwargs)
                return executor

            executor_func = await make_executor(tool_name)

            registry.register(
                name=tool_name,
                description=f"[{self.name}] {description}",
                parameters=input_schema,
                func=executor_func,
            )
            registered_names.append(tool_name)
            logger.info(f"🔌 Registered MCP tool '{tool_name}' from server '{self.name}'")

        return registered_names

    async def close(self):
        """Gracefully closes pipes and shuts down the subprocess to prevent zombies."""
        logger.info(f"🛑 Shutting down MCP server '{self.name}'...")
        if self._reader_task:
            self._reader_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

        self.process = None
        logger.info(f"MCP server '{self.name}' stopped.")

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional
from app.llm import format_for_whatsapp, DEFAULT_SYSTEM_PROMPT, get_system_prompt

logger = logging.getLogger("HermesAgent")


class ToolRegistry:
    """
    Registry that stores tool functions and their JSON schemas for LLMs.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        """
        Registers a Python function as a tool that the agent can call.
        """
        self._tools[name] = func
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }
        self._schemas.append(schema)
        logger.info(f"Registered tool: {name}")

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns the list of OpenAI-compatible tool schemas for the LLM."""
        return self._schemas

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Executes a registered tool function safely.
        Resolves common tool aliases (e.g. read_web -> read_webpage).
        Returns the result as a string or an error message.
        """
        # Common tool aliases to prevent naming mismatch failures
        aliases = {
            "read_web": "read_webpage",
            "web_read": "read_webpage",
            "browse": "read_webpage",
            "browse_url": "read_webpage",
            "read_url": "read_webpage",
            "web_search": "search_web",
            "google_search": "search_web",
            "search": "search_web",
            "internet_search": "search_web",
        }
        actual_name = aliases.get(name, name)

        if actual_name not in self._tools:
            return f"Error: Tool '{name}' is not registered."

        func = self._tools[actual_name]
        try:
            # Check if function is async or sync
            import inspect
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
            return f"Error executing {name}: {str(e)}"


# Global default registry
default_registry = ToolRegistry()


class HermesAgent:
    """
    The Hermes Agent Orchestrator.
    Manages the ReAct reasoning loop (Reason -> Act -> Observe -> Finish).
    """
    def __init__(self, registry: Optional[ToolRegistry] = None, max_steps: int = 5):
        self.registry = registry or default_registry
        self.max_steps = max_steps

    async def run(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Runs the autonomous agent loop:
        1. Packs history + user message.
        2. Calls the LLM with available tools.
        3. If LLM requests tool execution, runs the tool and feeds result back.
        4. Stops when LLM produces a final response or reaches max_steps.
        """
        from app.llm import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL
        from groq import AsyncGroq

        # Prepare messages payload with dynamic time-anchored system prompt
        active_prompt = system_prompt or get_system_prompt()
        messages = [{"role": "system", "content": active_prompt}]
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        messages.append({"role": "user", "content": user_text})

        tools = self.registry.get_schemas()
        client = AsyncGroq(api_key=GROQ_API_KEY)

        step = 0
        while step < self.max_steps:
            step += 1
            print(f"🤖 [Hermes Agent] Step {step}/{self.max_steps}...", flush=True)

            try:
                kwargs = {
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1000,
                }
                # Attach tools only if tools are registered
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = await client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message

                # CASE 1: LLM decided to call one or more tools
                if message.tool_calls:
                    print(f"🔧 [Hermes Agent] LLM requested {len(message.tool_calls)} tool call(s)", flush=True)
                    
                    # Append assistant's decision to messages history
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })

                    # Execute each tool requested by the LLM
                    for tc in message.tool_calls:
                        func_name = tc.function.name
                        try:
                            func_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}

                        print(f"⚡ [Hermes Agent] Executing tool: {func_name}({func_args})", flush=True)
                        result = await self.registry.execute(func_name, func_args)
                        print(f"📄 [Hermes Agent] Tool result: {result[:120]}...", flush=True)

                        # Feed the observation / tool result back to LLM
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": func_name,
                            "content": result
                        })

                    # Loop continues! LLM will inspect tool output in next iteration.
                    continue

                # CASE 2: LLM has finished thinking and produced the final user response
                raw_text = message.content or ""
                return format_for_whatsapp(raw_text)

            except Exception as e:
                err_str = str(e)
                # 1. Rate limit backoff
                if ("429" in err_str or "rate_limit" in err_str.lower()) and step < self.max_steps:
                    print(f"⏳ [Hermes Agent] Rate limit hit (429). Backing off for 10s and retrying... ({err_str[:60]})", flush=True)
                    await asyncio.sleep(10)
                    step -= 1
                    continue

                # 2. Auto-recovery from Groq 400 tool_use_failed
                if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                    print(f"🛠️ [Hermes Agent] Caught tool_use_failed (400). Attempting auto-recovery...", flush=True)
                    fn_name, fn_args = self._extract_failed_tool_call(e)
                    if fn_name:
                        print(f"⚡ [Hermes Agent Auto-Recovery] Extracted tool: {fn_name}({fn_args})", flush=True)
                        result = await self.registry.execute(fn_name, fn_args)
                        print(f"📄 [Hermes Agent Auto-Recovery] Tool result: {result[:120]}...", flush=True)

                        synthetic_call_id = f"recovered_call_{step}"
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": synthetic_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": fn_name,
                                        "arguments": json.dumps(fn_args)
                                    }
                                }
                            ]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": synthetic_call_id,
                            "name": fn_name,
                            "content": result
                        })
                        continue

                    # If extraction not possible, fall back to answering directly without tools
                    try:
                        print("🔄 [Hermes Agent Auto-Recovery] Completing directly without tools...", flush=True)
                        fallback_resp = await client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=1000
                        )
                        raw_text = fallback_resp.choices[0].message.content or ""
                        return format_for_whatsapp(raw_text)
                    except Exception as fb_err:
                        logger.error(f"Fallback generation also failed: {fb_err}")

                # 3. Clean human-friendly error response (NEVER leak raw 400 code to user)
                print(f"❌ [Hermes Agent] Error during execution: {e}", flush=True)
                return "⚠️ I ran into a brief hiccup while processing that. Could you try asking again or rephrasing?"

        return "⚠️ I reached the maximum step limit trying to process your request."

    def _extract_failed_tool_call(self, err: Exception) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Recovers tool call name and arguments when Groq raises tool_use_failed (400).
        Parses both Groq JSON and XML (<tool_call><function=...><parameter=...>) syntax.
        """
        failed_gen = ""
        if hasattr(err, "body") and isinstance(err.body, dict):
            failed_gen = str(err.body.get("error", {}).get("failed_generation", ""))

        err_str = str(err)
        if not failed_gen:
            idx = err_str.find("'failed_generation':")
            if idx == -1:
                idx = err_str.find('"failed_generation":')
            if idx != -1:
                snippet = err_str[idx + len("'failed_generation':"):].strip()
                if (snippet.startswith("'") and snippet.endswith("'")) or (snippet.startswith('"') and snippet.endswith('"')):
                    failed_gen = snippet[1:-1]
                else:
                    failed_gen = snippet

        if not failed_gen:
            failed_gen = err_str

        # Pattern 1: Direct or embedded JSON: {"name": "...", "arguments": {...}}
        try:
            d = json.loads(failed_gen)
            if isinstance(d, dict) and "name" in d:
                args = d.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                return d["name"], args
        except Exception:
            pass

        json_match = re.search(r'(\{"name"\s*:\s*"[^"]+".*?\})', failed_gen, re.DOTALL)
        if json_match:
            try:
                d = json.loads(json_match.group(1))
                args = d.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                return d.get("name"), args
            except Exception:
                pass

        # Pattern 2: XML format: <function=name><parameter=key>val</parameter>
        fn_match = re.search(r'<function=([a-zA-Z0-9_-]+)>', failed_gen)
        if fn_match:
            fn_name = fn_match.group(1)
            params = {}
            for pm in re.finditer(r'<parameter=([a-zA-Z0-9_-]+)>(.*?)</parameter>', failed_gen, re.DOTALL):
                params[pm.group(1)] = pm.group(2).strip()
            return fn_name, params

        return None, {}

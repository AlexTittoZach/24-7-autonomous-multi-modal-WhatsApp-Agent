import asyncio
import json
import sys
import urllib.parse
from typing import Any, Dict, List, Optional
import httpx

USER_AGENT = "JarvisWhatsAppAgent/1.0 (Personal AI Assistant on AWS EC2)"

# JSON-RPC 2.0 Output Helper
def send_response(msg: Dict[str, Any]):
    """Writes a single JSON-RPC message line to stdout and flushes."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


# Cache to avoid repeating Nominatim calls
_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}

OSM_SYNONYMS = {
    "movie theatre": "cinema",
    "movie theater": "cinema",
    "theatre": "cinema",
    "movies": "cinema",
    "cinema": "cinema",
    "multiplex": "cinema",
    "petrol pump": "fuel",
    "gas station": "fuel",
    "fuel station": "fuel",
    "eatery": "restaurant",
    "food": "restaurant",
    "gym": "fitness_centre",
    "fitness center": "fitness_centre",
    "doctor": "clinic",
    "medical store": "pharmacy",
    "chemist": "pharmacy",
    "medical": "pharmacy",
    "hospital": "hospital",
    "atm": "atm",
    "bank": "bank",
}


# --- Tool 1: Geocode Address ---
async def geocode_address(address: str) -> Dict[str, Any]:
    """Converts a textual address into GPS coordinates using Nominatim with memory caching."""
    clean_addr = address.strip().lower()
    if clean_addr in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[clean_addr]

    encoded_addr = urllib.parse.quote(address)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_addr}&format=json&limit=1"
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return {"error": f"Nominatim API error: HTTP {resp.status_code}"}
        data = resp.json()
        if not data:
            return {"error": f"No location found for '{address}'"}
        
        result = data[0]
        res = {
            "display_name": result.get("display_name"),
            "lat": float(result.get("lat")),
            "lon": float(result.get("lon")),
            "type": result.get("type"),
        }
        _GEOCODE_CACHE[clean_addr] = res
        return res


# --- Tool 2: Reverse Geocode Coordinates ---
async def reverse_geocode(latitude: float, longitude: float) -> Dict[str, Any]:
    """Converts GPS latitude and longitude coordinates into a human-readable street address."""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json"
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return {"error": f"Nominatim API error: HTTP {resp.status_code}"}
        data = resp.json()
        if not data:
            return {"error": f"No address found for coordinates {latitude}, {longitude}"}
        
        return {
            "display_name": data.get("display_name"),
            "address": data.get("address", {}),
            "lat": float(data.get("lat", latitude)),
            "lon": float(data.get("lon", longitude)),
        }


# --- Tool 3: Search Nearby Places ---
async def search_nearby_places(query: str, location: str, limit: int = 5) -> str:
    """
    Searches for points of interest (cinemas, cafes, hospitals, petrol pumps, etc.)
    around a specified location using radial bounding-box search and taxonomy mapping.
    """
    osm_query = OSM_SYNONYMS.get(query.lower().strip(), query.strip())
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=12.0) as client:
        # Step 1: Geocode location to get coordinates center
        center_geo = await geocode_address(location) if location else {}
        data = []

        if "lat" in center_geo and "lon" in center_geo:
            lat = center_geo["lat"]
            lon = center_geo["lon"]
            # Radial box of ~10km (delta = 0.09 degrees)
            delta = 0.09
            box = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
            bounded_url = (
                f"https://nominatim.openstreetmap.org/search?"
                f"q={urllib.parse.quote(osm_query)}&viewbox={box}&bounded=1&format=json&limit={limit}"
            )
            resp = await client.get(bounded_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()

        # Step 2: Fallback to textual search if radial search returned no results
        if not data:
            search_text = f"{osm_query} in {location}" if location else osm_query
            encoded_query = urllib.parse.quote(search_text)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&addressdetails=1&limit={limit}"
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()

        if not data:
            return f"No results found for '{query}' in or around '{location}'."

        results = []
        for i, item in enumerate(data[:limit], 1):
            name = item.get("display_name", "Unknown Name")
            lat = item.get("lat")
            lon = item.get("lon")
            maps_link = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"
            results.append(
                f"{i}. *{name}*\n"
                f"   📍 Location: {lat}, {lon}\n"
                f"   🗺️ Map: {maps_link}"
            )

        return "\n\n".join(results)


# --- Tool 4: Distance and Driving Route ---
async def get_distance_and_route(start_location: str, destination: str) -> str:
    """Calculates driving distance and travel time between two locations using concurrent geocoding and OSRM."""
    # Step 1: Geocode start and destination concurrently
    start_geo, dest_geo = await asyncio.gather(
        geocode_address(start_location),
        geocode_address(destination)
    )

    if "error" in start_geo:
        return f"Could not find start location: {start_geo['error']}"
    if "error" in dest_geo:
        return f"Could not find destination: {dest_geo['error']}"

    lon1, lat1 = start_geo["lon"], start_geo["lat"]
    lon2, lat2 = dest_geo["lon"], dest_geo["lat"]

    # Step 2: OSRM Route API
    osrm_url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.get(osrm_url, headers=headers)
        if resp.status_code != 200:
            return f"Error calculating route: HTTP {resp.status_code}"

        data = resp.json()
        routes = data.get("routes", [])
        if not routes:
            return f"No driving route found between '{start_location}' and '{destination}'."

        route = routes[0]
        distance_km = round(route.get("distance", 0) / 1000.0, 1)
        duration_mins = round(route.get("duration", 0) / 60.0)

        hours = duration_mins // 60
        mins = duration_mins % 60
        time_str = f"{hours} hr {mins} mins" if hours > 0 else f"{mins} mins"

        return (
            f"🚗 *Driving Route Summary*\n"
            f"• *From*: {start_geo['display_name']}\n"
            f"• *To*: {dest_geo['display_name']}\n"
            f"• *Distance*: *{distance_km} km*\n"
            f"• *Estimated Time*: *{time_str}* (typical driving conditions)"
        )


# --- MCP Tool Schemas ---
TOOLS_DEFINITIONS = [
    {
        "name": "search_nearby_places",
        "description": "Searches for points of interest (cinemas, movie theatres, cafes, hospitals, restaurants, ATMs, pharmacies, etc.) in or near a location using OpenStreetMap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The category or name of place to find, e.g. 'movie theatre', 'cinema', 'cafe', 'hospital', 'petrol pump'",
                },
                "location": {
                    "type": "string",
                    "description": "The neighborhood, area, or city, e.g. 'Marine Drive, Kochi', 'Edappally', 'MG Road, Bangalore'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                },
            },
            "required": ["query", "location"],
        },
    },
    {
        "name": "geocode_address",
        "description": "Converts any address, city, or landmark into GPS latitude/longitude coordinates using OpenStreetMap Nominatim.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "The full or partial address/place name to geocode",
                },
            },
            "required": ["address"],
        },
    },
    {
        "name": "get_distance_and_route",
        "description": "Calculates estimated driving distance (in km) and travel time (in minutes) between two locations using OpenStreetMap routing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_location": {
                    "type": "string",
                    "description": "Starting address, city, or landmark name",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination address, city, or landmark name",
                },
            },
            "required": ["start_location", "destination"],
        },
    },
    {
        "name": "reverse_geocode",
        "description": "Converts GPS latitude and longitude coordinates into a human-readable street address and neighborhood name using OpenStreetMap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate, e.g. 9.9794",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude coordinate, e.g. 76.2768",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
]


# --- Main MCP Protocol Event Loop ---
async def handle_mcp_request(line: str):
    """Parses incoming JSON-RPC 2.0 requests from stdin and sends response to stdout."""
    line = line.strip()
    if not line:
        return

    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        return

    method = req.get("method")
    req_id = req.get("id")

    # 1. Handshake: initialize
    if method == "initialize":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "openstreetmap-mcp-server",
                    "version": "1.0.0"
                }
            }
        })

    # 2. Handshake confirmation: notifications/initialized
    elif method == "notifications/initialized":
        pass  # Notification requires no response

    # 3. Discovery: tools/list
    elif method == "tools/list":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS_DEFINITIONS
            }
        })

    # 4. Execution: tools/call
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            if tool_name == "search_nearby_places":
                output = await search_nearby_places(
                    query=args.get("query", ""),
                    location=args.get("location", ""),
                    limit=args.get("limit", 5)
                )
            elif tool_name == "geocode_address":
                geo = await geocode_address(address=args.get("address", ""))
                output = json.dumps(geo, indent=2)
            elif tool_name == "get_distance_and_route":
                output = await get_distance_and_route(
                    start_location=args.get("start_location", ""),
                    destination=args.get("destination", "")
                )
            elif tool_name == "reverse_geocode":
                geo = await reverse_geocode(
                    latitude=float(args.get("latitude", 0)),
                    longitude=float(args.get("longitude", 0))
                )
                output = json.dumps(geo, indent=2)
            else:
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method/Tool '{tool_name}' not found"
                    }
                })
                return

            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(output)
                        }
                    ]
                }
            })

        except Exception as e:
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error executing {tool_name}: {str(e)}"
                        }
                    ],
                    "isError": True
                }
            })

    # 5. Ping / health check
    elif method == "ping":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        })


async def main():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        await handle_mcp_request(line.decode("utf-8"))


if __name__ == "__main__":
    asyncio.run(main())

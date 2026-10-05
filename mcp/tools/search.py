import logging
from typing import Dict, Any

logger = logging.getLogger("SearchTool")


def search_web(query: str, max_results: int = 6) -> str:
    """
    Searches the live web using DuckDuckGo and returns top snippets and source links.
    """
    from ddgs import DDGS

    query_clean = query.strip()
    if not query_clean:
        return "Error: Empty search query."

    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(query_clean, max_results=max_results))
        
        if not raw_results:
            return f"No search results found for: '{query_clean}'."

        formatted_results = []
        for i, r in enumerate(raw_results, 1):
            title = r.get("title", "No Title")
            body = r.get("body", "").strip()
            href = r.get("href", "")
            formatted_results.append(f"{i}. *{title}*\n   {body}\n   🔗 {href}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"DuckDuckGo search error for '{query_clean}': {e}", exc_info=True)
        return f"Unable to fetch live search results right now: {str(e)}"


# -------------------------------------------------------------------
# Tool Schema for the LLM (Tells the AI when to browse the web)
# -------------------------------------------------------------------

SEARCH_WEB_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search keywords, e.g. 'weather in Kochi today', 'latest tech headlines', 'cricket match score'"
        }
    },
    "required": ["query"]
}

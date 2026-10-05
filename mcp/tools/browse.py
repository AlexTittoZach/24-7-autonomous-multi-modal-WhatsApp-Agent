import re
import logging
from typing import Dict, Any
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("BrowseTool")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def read_webpage(url: str, max_chars: int = 3500) -> str:
    """
    Fetches a webpage by URL, removes HTML junk (scripts, ads, navigation),
    and returns the clean text content for the AI to read and analyze.
    """
    url_clean = url.strip()
    if not url_clean.startswith("http://") and not url_clean.startswith("https://"):
        url_clean = "https://" + url_clean

    try:
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url_clean)

        if resp.status_code != 200:
            return f"Failed to fetch webpage. HTTP Status code: {resp.status_code}"

        # Parse HTML
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip away unwanted tags (scripts, styles, navbars, footers)
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            tag.decompose()

        # Extract text and collapse excessive whitespace
        raw_text = soup.get_text(separator="\n", strip=True)
        clean_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        clean_text = "\n".join(clean_lines)

        if not clean_text:
            return f"Webpage at {url_clean} returned no readable text."

        # Cap text length to avoid overflowing AI token budget
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "\n\n... [Content truncated for length]"

        return f"--- Content from {url_clean} ---\n\n{clean_text}"

    except Exception as e:
        logger.error(f"Error reading webpage '{url_clean}': {e}", exc_info=True)
        return f"Unable to read webpage at {url_clean}: {str(e)}"


# -------------------------------------------------------------------
# Tool Schema for the LLM (Tells the AI when to read a URL)
# -------------------------------------------------------------------

READ_WEBPAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The exact full URL of the webpage to read and analyze, e.g. 'https://hermes-agent.org/' or 'https://en.wikipedia.org/wiki/Python'"
        }
    },
    "required": ["url"]
}

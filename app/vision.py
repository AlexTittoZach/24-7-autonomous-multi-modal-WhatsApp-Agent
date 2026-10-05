import os
import base64
import httpx
from dotenv import load_dotenv
from app.llm import format_for_whatsapp

load_dotenv()

VISION_PROVIDER = os.getenv("VISION_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")

BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY", "")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-south-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")


def get_image_format(mime_type: str) -> str:
    """Converts standard MIME types into Bedrock-accepted format string."""
    mime = (mime_type or "").lower()
    if "png" in mime:
        return "png"
    elif "gif" in mime:
        return "gif"
    elif "webp" in mime:
        return "webp"
    return "jpeg"


async def analyze_image_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg", user_prompt: str = "") -> str:
    """
    Invokes Google Gemini Vision (e.g. Gemini 3.6 Flash) with high-res OCR and multimodal comprehension.
    """
    if not GEMINI_API_KEY:
        return "❌ Error: `GEMINI_API_KEY` is not set in `.env`."

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    actual_mime = mime_type if mime_type and "/" in mime_type else "image/jpeg"

    prompt_text = user_prompt.strip() if user_prompt else (
        "Please analyze this image in detail. Identify and describe what is visible, "
        "extract any text, numbers, receipts, or data if present, "
        "and summarize key takeaways formatted cleanly for WhatsApp (*bold*)."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": actual_mime,
                            "data": b64_image
                        }
                    }
                ]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                raw_reply = data["candidates"][0]["content"]["parts"][0]["text"]
                return format_for_whatsapp(raw_reply)
            else:
                print(f"❌ Gemini Vision API Error ({resp.status_code}): {resp.text}", flush=True)
                return f"⚠️ Gemini Vision returned HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            print(f"❌ Connection error to Gemini Vision: {e}", flush=True)
            return f"⚠️ Failed to communicate with Gemini Vision engine: {e}"


async def analyze_image_with_bedrock(image_bytes: bytes, mime_type: str = "image/jpeg", user_prompt: str = "") -> str:
    """
    Analyzes an image using AWS Bedrock Converse API with native multimodal capabilities.
    Supports Amazon Nova Lite, Nova Pro, Claude 3.5 Sonnet, etc.
    """
    if not BEDROCK_API_KEY:
        return "❌ Error: `BEDROCK_API_KEY` is not set in `.env`."

    img_format = get_image_format(mime_type)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt_text = user_prompt.strip() if user_prompt else (
        "Please analyze this image in detail. "
        "Describe what is shown, extract any text, data, numbers, or receipts visible, "
        "and provide helpful insights formatted cleanly for WhatsApp (*bold*)."
    )

    endpoint = f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/model/{BEDROCK_MODEL_ID}/converse"
    headers = {
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": img_format,
                            "source": {
                                "bytes": b64_image
                            }
                        }
                    },
                    {
                        "text": prompt_text
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 1500,
            "temperature": 0.4
        }
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            resp = await client.post(endpoint, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                raw_reply = data["output"]["message"]["content"][0]["text"]
                return format_for_whatsapp(raw_reply)
            else:
                print(f"❌ Bedrock Vision API Error ({resp.status_code}): {resp.text}", flush=True)
                return (
                    f"⚠️ Bedrock Vision returned HTTP {resp.status_code}.\n"
                    f"Details: {resp.text}"
                )
        except Exception as e:
            print(f"❌ Connection error to Bedrock Vision: {e}", flush=True)
            return f"⚠️ Failed to communicate with Bedrock Vision engine: {e}"


async def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg", user_prompt: str = "") -> str:
    """
    Unified entry point for image vision analysis.
    Defaults to Gemini if configured, or Bedrock if requested.
    """
    if VISION_PROVIDER == "bedrock" and BEDROCK_API_KEY:
        print(f"👁️ Routing image to AWS Bedrock ({BEDROCK_MODEL_ID})...", flush=True)
        reply = await analyze_image_with_bedrock(image_bytes, mime_type, user_prompt)
        # If Bedrock has account lock, fall back gracefully to Gemini if key exists
        if "Operation not allowed" in reply and GEMINI_API_KEY:
            print("⚠️ Bedrock account locked, seamlessly falling back to Gemini Vision...", flush=True)
            return await analyze_image_with_gemini(image_bytes, mime_type, user_prompt)
        return reply
    else:
        print(f"⚡ Routing image to Google Gemini Vision ({GEMINI_VISION_MODEL})...", flush=True)
        return await analyze_image_with_gemini(image_bytes, mime_type, user_prompt)

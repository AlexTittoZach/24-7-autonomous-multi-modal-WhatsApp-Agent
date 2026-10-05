import os
import httpx
from dotenv import load_dotenv

load_dotenv()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
GRAPH_API_VERSION = "v20.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"


async def send_whatsapp_message(to: str, text: str) -> dict:
    """
    Sends an outbound text message to a WhatsApp user via Meta Cloud API.
    
    Args:
        to: Recipient phone number with country code (e.g. "91XXXXXXXXXX")
        text: The text message content to send
        
    Returns:
        dict: Meta API JSON response
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("❌ Error: WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID not set in .env", flush=True)
        return {"error": "Missing credentials"}

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(BASE_URL, headers=headers, json=payload)
            response_data = response.json()
            if response.status_code == 200:
                msg_id = response_data.get("messages", [{}])[0].get("id")
                print(f"📤 Message successfully sent to {to} | Message ID: {msg_id}", flush=True)
            else:
                print(f"❌ Failed to send message to {to}: HTTP {response.status_code} | {response_data}", flush=True)
            return response_data
        except Exception as e:
            print(f"❌ Exception sending WhatsApp message: {repr(e)}", flush=True)
            return {"error": repr(e)}


async def mark_message_as_read(message_id: str) -> dict:
    """
    Marks an incoming WhatsApp message as read (sends blue ticks to the user).
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        return {"error": "Missing credentials"}

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.post(BASE_URL, headers=headers, json=payload)
            return response.json()
        except Exception as e:
            print(f"⚠️ Exception marking message as read: {e}", flush=True)
            return {"error": str(e)}


async def download_whatsapp_media(media_id: str) -> bytes:
    """
    Downloads media (audio, images, documents) from Meta Cloud API using media_id.
    
    Step 1: Queries Graph API for the temporary media URL.
    Step 2: Downloads the raw file bytes using our WHATSAPP_TOKEN.
    """
    if not WHATSAPP_TOKEN:
        print("❌ Error: WHATSAPP_TOKEN not configured.", flush=True)
        return b""

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "User-Agent": "curl/7.64.1",  # Meta requires a valid User-Agent
    }

    # Step 1: Retrieve the download URL from Meta
    media_info_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            info_resp = await client.get(media_info_url, headers=headers)
            if info_resp.status_code != 200:
                print(f"❌ Failed to get media info for ID {media_id}: HTTP {info_resp.status_code} | {info_resp.text}", flush=True)
                return b""

            download_url = info_resp.json().get("url")
            if not download_url:
                print(f"❌ No download URL found in Meta response for {media_id}", flush=True)
                return b""

            # Step 2: Download the raw audio/media bytes
            media_resp = await client.get(download_url, headers=headers)
            if media_resp.status_code == 200:
                print(f"📥 Successfully downloaded media {media_id} ({len(media_resp.content)} bytes)", flush=True)
                return media_resp.content
            else:
                print(f"❌ Failed to download media bytes: HTTP {media_resp.status_code}", flush=True)
                return b""

        except Exception as e:
            print(f"❌ Exception downloading media {media_id}: {e}", flush=True)
            return b""

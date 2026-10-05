import asyncio
import json
import os
import sys
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from agent.core import HermesAgent, default_registry
from mcp.client import StdioMCPClient
from mcp.tools.memory import (
    SAVE_MEMORY_SCHEMA,
    SEARCH_MEMORY_SCHEMA,
    save_memory,
    search_memory,
)
from mcp.tools.search import SEARCH_WEB_SCHEMA, search_web
from mcp.tools.browse import READ_WEBPAGE_SCHEMA, read_webpage
from mcp.tools.reminder import (
    SCHEDULE_REMINDER_SCHEMA,
    LIST_REMINDERS_SCHEMA,
    CANCEL_REMINDER_SCHEMA,
    schedule_reminder,
    list_reminders,
    cancel_reminder,
    get_due_reminders,
    mark_reminder_sent,
    advance_reminder_recurrence,
)
from app.session import clear_history, get_history, save_message
from app.whatsapp_client import mark_message_as_read, send_whatsapp_message, download_whatsapp_media
from app.transcriber import transcribe_audio
from app.document import extract_document_content
from app.vision import analyze_image

# Load variables from .env
load_dotenv()
VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "whatsapp_agent_super_secret_token_2026")
ALLOWED_NUMBERS = [n.strip() for n in os.getenv("ALLOWED_PHONE_NUMBERS", "").split(",") if n.strip()]

# Register MCP Tools with Hermes
default_registry.register(
    name="save_memory",
    description="Permanently save or update a user's personal fact, credential, date, or preference in memory.",
    parameters=SAVE_MEMORY_SCHEMA,
    func=save_memory
)

default_registry.register(
    name="search_memory",
    description="Search permanent memories for a user's saved facts, details, credentials, or preferences.",
    parameters=SEARCH_MEMORY_SCHEMA,
    func=search_memory
)

default_registry.register(
    name="search_web",
    description="Search the live internet for current events, breaking news, live weather, sports scores, or real-time facts.",
    parameters=SEARCH_WEB_SCHEMA,
    func=search_web
)

default_registry.register(
    name="read_webpage",
    description="Open, download, and read the full text content of a specific webpage URL to summarize or analyze it.",
    parameters=READ_WEBPAGE_SCHEMA,
    func=read_webpage
)

# Register aliases so LLMs calling 'read_web' or 'web_search' succeed directly with Groq
default_registry.register(
    name="read_web",
    description="Alias for read_webpage. Read full content of a webpage URL.",
    parameters=READ_WEBPAGE_SCHEMA,
    func=read_webpage
)

default_registry.register(
    name="web_search",
    description="Alias for search_web. Search the internet for live info.",
    parameters=SEARCH_WEB_SCHEMA,
    func=search_web
)

default_registry.register(
    name="schedule_reminder",
    description="Schedules a future reminder or alarm. Takes remind_at_iso in ISO format (YYYY-MM-DDTHH:MM:SS) calculated from current local time in your prompt, and reminder_text.",
    parameters=SCHEDULE_REMINDER_SCHEMA,
    func=schedule_reminder
)

default_registry.register(
    name="list_reminders",
    description="Lists all active pending reminders scheduled by the user.",
    parameters=LIST_REMINDERS_SCHEMA,
    func=list_reminders
)

default_registry.register(
    name="cancel_reminder",
    description="Cancels an active pending reminder by its numerical ID.",
    parameters=CANCEL_REMINDER_SCHEMA,
    func=cancel_reminder
)

# Initialize the Hermes Agent
agent = HermesAgent(registry=default_registry)


async def reminder_scheduler_loop():
    """
    Autonomous background worker that polls reminders.db every 15 seconds.
    Whenever a reminder is due, sends a proactive WhatsApp message to the user!
    """
    print("⏰ [Scheduler] Background reminder worker started.", flush=True)
    while True:
        try:
            due_reminders = get_due_reminders()
            for r in due_reminders:
                rem_id = r["id"]
                phone = r["phone_number"]
                text = r["reminder_text"]
                recurrence = r.get("recurrence", "none")
                print(f"⏰ [Scheduler] Firing due reminder #{rem_id} (recurrence: {recurrence}) to {phone}: '{text}'", flush=True)

                reminder_msg = f"⏰ *Reminder from Jarvis:*\n\n{text}"
                await send_whatsapp_message(to=phone, text=reminder_msg)
                save_message(phone, "assistant", reminder_msg)

                # If recurring, advance to next cycle; if one-off, mark as sent
                if recurrence in ["daily", "weekly"]:
                    advance_reminder_recurrence(rem_id, r["remind_at"], recurrence)
                else:
                    mark_reminder_sent(rem_id)
        except asyncio.CancelledError:
            print("⏰ [Scheduler] Background worker received stop signal.", flush=True)
            break
        except Exception as e:
            print(f"⚠️ [Scheduler Error]: {e}", flush=True)

        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Start background reminder loop
    scheduler_task = asyncio.create_task(reminder_scheduler_loop())

    # 2. Launch external OpenStreetMap MCP Server over stdio pipes
    osm_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp", "servers", "osm_server.py"))
    mcp_client = StdioMCPClient(
        command=[sys.executable, osm_server_path],
        name="OpenStreetMap"
    )
    try:
        await mcp_client.start()
        registered = await mcp_client.register_tools(default_registry)
        print(f"🗺️ [MCP] Connected to OpenStreetMap Server! Registered {len(registered)} tools: {registered}", flush=True)
    except Exception as e:
        print(f"⚠️ [MCP Error] Failed to connect to OpenStreetMap server: {e}", flush=True)
        mcp_client = None

    yield

    # Shutdown: Safely stop MCP server process and reminder loop
    if mcp_client:
        await mcp_client.close()

    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="WhatsApp AI Agent", lifespan=lifespan)


async def process_user_message(sender_phone: str, message_id: str, sender_name: str, text_body: str):
    """
    Background worker that processes incoming WhatsApp user text with Hermes Agent,
    allowing it to use MCP tools and session memory.
    """
    # 1. Mark message as read (sends blue ticks to the user immediately)
    if message_id:
        await mark_message_as_read(message_id)

    # 2. Security Check: Whitelist verification (if configured in .env)
    if ALLOWED_NUMBERS and sender_phone not in ALLOWED_NUMBERS:
        print(f"🚫 Unauthorized message from {sender_phone}. Ignored.", flush=True)
        return

    # 3. Handle Special Commands (e.g. /clear, /reset)
    cleaned_text = text_body.strip().lower()
    if cleaned_text in ["/clear", "/reset", "clear chat", "reset chat"]:
        clear_history(sender_phone)
        reply = "🧹 *Chat history cleared!* Starting a brand new conversation session. How can I help you today?"
        await send_whatsapp_message(to=sender_phone, text=reply)
        return

    # 4. Load past conversation history from SQLite (last 10 turns)
    history = get_history(sender_phone, limit=5)
    print(f"🧠 Loaded {len(history)} past messages for {sender_name} ({sender_phone})", flush=True)

    # 5. Run through Hermes Agent (Autonomously reasons, uses tools, and generates reply)
    print(f"🤖 [Hermes Agent] Processing message from {sender_name}: \"{text_body}\"", flush=True)
    ai_reply = await agent.run(user_text=text_body, conversation_history=history)

    # 6. Save this conversation turn into SQLite
    save_message(sender_phone, "user", text_body)
    save_message(sender_phone, "assistant", ai_reply)

    # 7. Send AI response back to WhatsApp via Meta Graph API
    await send_whatsapp_message(to=sender_phone, text=ai_reply)


async def process_user_audio(sender_phone: str, message_id: str, sender_name: str, media_id: str):
    """
    Background worker that handles incoming WhatsApp voice notes:
    1. Marks message as read (blue ticks).
    2. Downloads the .ogg audio file from Meta.
    3. Transcribes speech into text using Groq Whisper (0.3s).
    4. Hands transcribed text to Hermes Agent.
    5. Sends the final answer back to WhatsApp.
    """
    # 1. Mark message as read (sends blue ticks)
    if message_id:
        await mark_message_as_read(message_id)

    # 2. Security Check: Whitelist verification
    if ALLOWED_NUMBERS and sender_phone not in ALLOWED_NUMBERS:
        print(f"🚫 Unauthorized audio message from {sender_phone}. Ignored.", flush=True)
        return

    # 3. Download the audio file from Meta
    print(f"📥 Downloading voice note for {sender_name} (media_id: {media_id})...", flush=True)
    audio_bytes = await download_whatsapp_media(media_id)
    if not audio_bytes:
        await send_whatsapp_message(to=sender_phone, text="⚠️ Sorry, I could not download your voice note from WhatsApp.")
        return

    # 4. Transcribe audio with Groq Whisper
    print(f"🎙️ Transcribing voice note with Whisper...", flush=True)
    transcribed_text = await transcribe_audio(audio_bytes, filename="voice.ogg")
    if not transcribed_text:
        await send_whatsapp_message(to=sender_phone, text="⚠️ Sorry, I could not transcribe your audio message. Please try speaking clearly.")
        return

    print(f"🗣️ Transcribed Voice from {sender_name}: \"{transcribed_text}\"", flush=True)

    # 5. Load past conversation history from SQLite
    history = get_history(sender_phone, limit=5)

    # 6. Run through Hermes Agent
    print(f"🤖 [Hermes Agent] Processing transcribed query: \"{transcribed_text}\"", flush=True)
    ai_reply = await agent.run(user_text=transcribed_text, conversation_history=history)

    # 7. Save conversation turn to SQLite (noting it was a voice note)
    save_message(sender_phone, "user", f"🎙️ [Voice Note]: {transcribed_text}")
    save_message(sender_phone, "assistant", ai_reply)

    # 8. Send AI response back to WhatsApp
    await send_whatsapp_message(to=sender_phone, text=ai_reply)


async def process_user_document(
    sender_phone: str,
    message_id: str,
    sender_name: str,
    media_id: str,
    filename: str,
    caption: str = ""
):
    """
    Background worker that handles incoming WhatsApp documents (PDF, TXT, CSV, etc.):
    1. Marks message as read (blue ticks).
    2. Downloads the document file from Meta.
    3. Extracts text using pypdf / app.document.
    4. Handles scanned or password-protected files gracefully.
    5. Hands extracted document context to Hermes Agent.
    6. Sends AI response back to WhatsApp.
    """
    # 1. Mark message as read (sends blue ticks)
    if message_id:
        await mark_message_as_read(message_id)

    # 2. Security Check: Whitelist verification
    if ALLOWED_NUMBERS and sender_phone not in ALLOWED_NUMBERS:
        print(f"🚫 Unauthorized document from {sender_phone}. Ignored.", flush=True)
        return

    # 3. Download the document file from Meta
    print(f"📥 Downloading document '{filename}' for {sender_name} (media_id: {media_id})...", flush=True)
    file_bytes = await download_whatsapp_media(media_id)
    if not file_bytes:
        await send_whatsapp_message(to=sender_phone, text=f"⚠️ Sorry, I could not download *{filename}* from WhatsApp.")
        return

    # 4. If an image file was sent via the Document picker, route directly to Vision
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
    if filename.lower().endswith(IMAGE_EXTENSIONS):
        print(f"📸 Document '{filename}' is an image! Redirecting to process_user_image...", flush=True)
        await process_user_image(
            sender_phone=sender_phone,
            message_id=message_id,
            sender_name=sender_name,
            media_id=media_id,
            mime_type="image/jpeg",
            caption=caption,
        )
        return

    # 5. Extract document content
    print(f"📄 Extracting text from '{filename}' ({len(file_bytes)} bytes)...", flush=True)
    doc_res = extract_document_content(file_bytes, filename=filename)
    status = doc_res["status"]

    if status == "scanned_or_empty":
        notice = (
            f"📄 I received your document *{filename}* ({doc_res.get('page_count', 1)} page(s)), "
            f"but it appears to be a scanned document or image-only file with no selectable digital text.\n\n"
            f"💡 *Tip*: Please upload a digital PDF with selectable text, or send the pages as photos so I can inspect them!"
        )
        await send_whatsapp_message(to=sender_phone, text=notice)
        return

    if status == "encrypted":
        await send_whatsapp_message(
            to=sender_phone,
            text=f"🔒 The document *{filename}* is password-protected. Please remove the password and re-upload."
        )
        return

    if status == "unsupported":
        await send_whatsapp_message(
            to=sender_phone,
            text=f"⚠️ {doc_res.get('error_message', 'Unsupported document format.')}"
        )
        return

    if status != "success" or not doc_res.get("text"):
        await send_whatsapp_message(
            to=sender_phone,
            text=f"⚠️ Could not read document *{filename}*: {doc_res.get('error_message', 'Unknown parsing error.')}"
        )
        return

    extracted_text = doc_res["text"]
    page_count = doc_res.get("page_count", 1)
    print(f"📄 Successfully extracted {len(extracted_text)} chars from {filename} ({page_count} pages)", flush=True)

    # 5. Build contextual prompt for Hermes Agent
    if caption.strip():
        user_prompt = (
            f"📄 [Uploaded Document: {filename} ({page_count} pages)]\n\n"
            f"User Request: {caption.strip()}\n\n"
            f"Document Content:\n{extracted_text}"
        )
        display_text = f"📄 [{filename}]: {caption.strip()}"
    else:
        user_prompt = (
            f"📄 [Uploaded Document: {filename} ({page_count} pages)]\n\n"
            f"Please provide a clear, well-structured summary of this document, highlight its key takeaways, "
            f"and let me know how you can assist me further with it.\n\n"
            f"Document Content:\n{extracted_text}"
        )
        display_text = f"📄 [Uploaded Document: {filename}]"

    # 6. Load past conversation history from SQLite
    history = get_history(sender_phone, limit=5)

    # 7. Run through Hermes Agent
    print(f"🤖 [Hermes Agent] Processing document '{filename}' with prompt...", flush=True)
    ai_reply = await agent.run(user_text=user_prompt, conversation_history=history)

    # 8. Save conversation turn to SQLite
    save_message(sender_phone, "user", display_text)
    save_message(sender_phone, "assistant", ai_reply)

    # 9. Send AI response back to WhatsApp
    await send_whatsapp_message(to=sender_phone, text=ai_reply)


async def process_user_image(
    sender_phone: str,
    message_id: str,
    sender_name: str,
    media_id: str,
    mime_type: str = "image/jpeg",
    caption: str = ""
):
    """
    Background worker that handles incoming WhatsApp images/photos:
    1. Marks message as read (blue ticks).
    2. Downloads raw image bytes from Meta CDN.
    3. Runs visual multimodal reasoning via AWS Bedrock Converse API (Nova Lite).
    4. Saves the interaction to SQLite chat history.
    5. Sends AI response back to the user on WhatsApp.
    """
    if message_id:
        await mark_message_as_read(message_id)

    if ALLOWED_NUMBERS and sender_phone not in ALLOWED_NUMBERS:
        print(f"🚫 Unauthorized image from {sender_phone}. Ignored.", flush=True)
        return

    print(f"📸 Downloading image for {sender_name} (media_id: {media_id})...", flush=True)
    image_bytes = await download_whatsapp_media(media_id)
    if not image_bytes:
        await send_whatsapp_message(
            to=sender_phone,
            text="⚠️ Sorry, I could not download your image from WhatsApp."
        )
        return

    print(f"👁️ Analyzing image with Bedrock Vision (caption: '{caption}')...", flush=True)
    vision_reply = await analyze_image(
        image_bytes=image_bytes,
        mime_type=mime_type,
        user_prompt=caption
    )

    # Save to SQLite history
    user_label = f"📸 [Photo]: {caption}" if caption else "📸 [Uploaded a photo]"
    save_message(sender_phone, "user", user_label)
    save_message(sender_phone, "assistant", vision_reply)

    # Send response back to WhatsApp
    await send_whatsapp_message(to=sender_phone, text=vision_reply)


async def process_user_location(
    sender_phone: str,
    message_id: str,
    sender_name: str,
    latitude: float,
    longitude: float,
    name: str = "",
    address: str = ""
):
    """
    Background worker that handles incoming WhatsApp Location Pins:
    1. Marks message as read (blue ticks).
    2. Reverse-geocodes GPS coordinates into human address using OpenStreetMap Nominatim.
    3. Feeds location context to Hermes Agent.
    4. Automatically logs current location to SQLite history & memory.
    5. Sends helpful response to user on WhatsApp.
    """
    if message_id:
        await mark_message_as_read(message_id)

    if ALLOWED_NUMBERS and sender_phone not in ALLOWED_NUMBERS:
        print(f"🚫 Unauthorized location from {sender_phone}. Ignored.", flush=True)
        return

    print(f"📍 Processing location pin for {sender_name}: ({latitude}, {longitude})", flush=True)

    # 1. Reverse-geocode coordinates via OpenStreetMap Nominatim
    display_address = address or name or f"{latitude}, {longitude}"
    url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json"
    headers = {"User-Agent": "JarvisWhatsAppAgent/1.0"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                display_address = data.get("display_name", display_address)
    except Exception as e:
        print(f"⚠️ Reverse geocode error: {e}", flush=True)

    # 2. Formulate prompt for Hermes Agent
    prompt = (
        f"📍 [User dropped a live WhatsApp GPS Location Pin]\n"
        f"• GPS Coordinates: {latitude}, {longitude}\n"
        f"• Resolved Address: {display_address}\n\n"
        f"Please acknowledge receipt of this location warmly, confirm the neighborhood/city detected, "
        f"save this as their current location in memory if appropriate, and ask what you can find nearby for them "
        f"(e.g., cafes, restaurants, movie theatres, petrol pumps, ATMs, pharmacies, or driving routes)."
    )

    # 3. Load chat history
    history = get_history(sender_phone, limit=5)

    # 4. Run through Hermes Agent
    ai_reply = await agent.run(user_text=prompt, conversation_history=history)

    # 5. Save to chat history
    save_message(sender_phone, "user", f"📍 [GPS Pin]: {display_address}")
    save_message(sender_phone, "assistant", ai_reply)

    # 6. Send reply to WhatsApp
    await send_whatsapp_message(to=sender_phone, text=ai_reply)



@app.get("/")
async def root():
    return {"status": "WhatsApp agent backend is running"}


# 1. Verification Handshake Endpoint (GET)
@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    """Handles Meta's initial verification challenge."""
    if mode == "subscribe" and token == VERIFY_TOKEN:
        print(f"✅ Webhook verified successfully! Challenge: {challenge}", flush=True)
        return PlainTextResponse(content=challenge, status_code=200)

    print("❌ Webhook verification failed! Invalid token or mode.", flush=True)
    raise HTTPException(status_code=403, detail="Verification failed")


# 2. Inbound Message Handler (POST)
@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives incoming messages and events from Meta."""
    data = await request.json()
    print("\n📥 [RAW INCOMING PAYLOAD FROM META]:", flush=True)
    print(json.dumps(data, indent=2), flush=True)

    # Check if this is a WhatsApp message event
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        # 1. If it's an incoming user message
        if "messages" in value:
            msg = value["messages"][0]
            message_id = msg.get("id")
            contact = value.get("contacts", [{}])[0]
            sender_name = contact.get("profile", {}).get("name", "there")
            sender_phone = msg.get("from")
            msg_type = msg.get("type", "unknown")

            print("\n" + "=" * 50, flush=True)
            print("📩 NEW WHATSAPP MESSAGE RECEIVED!", flush=True)
            print(f"👤 Sender Name  : {sender_name}", flush=True)
            print(f"📱 Sender Phone : {sender_phone}", flush=True)
            print(f"📦 Message Type : {msg_type}", flush=True)

            if msg_type == "text":
                text_body = msg.get("text", {}).get("body", "")
                print(f"💬 Message Body : {text_body}", flush=True)
                
                # Queue the reply to run in the background
                if sender_phone:
                    background_tasks.add_task(
                        process_user_message,
                        sender_phone=sender_phone,
                        message_id=message_id,
                        sender_name=sender_name,
                        text_body=text_body,
                    )

            elif msg_type == "audio":
                audio_info = msg.get("audio", {})
                media_id = audio_info.get("id")
                print(f"🎙️ Voice Note Media ID: {media_id}", flush=True)

                # Queue the audio downloader + Whisper transcription in the background
                if sender_phone and media_id:
                    background_tasks.add_task(
                        process_user_audio,
                        sender_phone=sender_phone,
                        message_id=message_id,
                        sender_name=sender_name,
                        media_id=media_id,
                    )

            elif msg_type == "document":
                doc_info = msg.get("document", {})
                media_id = doc_info.get("id")
                filename = doc_info.get("filename", "document.pdf")
                mime_type = doc_info.get("mime_type", "")
                caption = doc_info.get("caption", "")
                print(f"📄 Document received: {filename} (Media ID: {media_id}, Caption: '{caption}')", flush=True)

                IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
                if filename.lower().endswith(IMAGE_EXTENSIONS) or mime_type.startswith("image/"):
                    print(f"📸 Document is an image ({filename})! Routing directly to Vision engine...", flush=True)
                    if sender_phone and media_id:
                        background_tasks.add_task(
                            process_user_image,
                            sender_phone=sender_phone,
                            message_id=message_id,
                            sender_name=sender_name,
                            media_id=media_id,
                            mime_type=mime_type or "image/jpeg",
                            caption=caption,
                        )
                else:
                    if sender_phone and media_id:
                        background_tasks.add_task(
                            process_user_document,
                            sender_phone=sender_phone,
                            message_id=message_id,
                            sender_name=sender_name,
                            media_id=media_id,
                            filename=filename,
                            caption=caption,
                        )

            elif msg_type == "image":
                image_info = msg.get("image", {})
                media_id = image_info.get("id")
                mime_type = image_info.get("mime_type", "image/jpeg")
                caption = image_info.get("caption", "")
                print(f"📸 Image received from {sender_name} (Media ID: {media_id}, Caption: '{caption}')", flush=True)

                if sender_phone and media_id:
                    background_tasks.add_task(
                        process_user_image,
                        sender_phone=sender_phone,
                        message_id=message_id,
                        sender_name=sender_name,
                        media_id=media_id,
                        mime_type=mime_type,
                        caption=caption,
                    )

            elif msg_type == "location":
                loc_info = msg.get("location", {})
                latitude = loc_info.get("latitude")
                longitude = loc_info.get("longitude")
                name = loc_info.get("name", "")
                address = loc_info.get("address", "")
                print(f"📍 Location pin received from {sender_name}: ({latitude}, {longitude}) - Name: '{name}', Address: '{address}'", flush=True)

                if sender_phone and latitude is not None and longitude is not None:
                    background_tasks.add_task(
                        process_user_location,
                        sender_phone=sender_phone,
                        message_id=message_id,
                        sender_name=sender_name,
                        latitude=latitude,
                        longitude=longitude,
                        name=name,
                        address=address,
                    )

            else:
                print(f"📎 Media/Other  : {msg.get(msg_type)}", flush=True)

            print("=" * 50 + "\n", flush=True)

        # 2. If it's a message status update (sent, delivered, read)
        elif "statuses" in value:
            status = value["statuses"][0]
            recipient = status.get("recipient_id", "Unknown")
            curr_status = status.get("status", "Unknown")
            print(f"ℹ️ Status Update: Message to {recipient} is '{curr_status}'", flush=True)

    except Exception as e:
        print(f"⚠️ Raw payload received (error parsing): {data} | Error: {e}", flush=True)

    return {"status": "received"}
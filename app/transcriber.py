import os
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


async def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg", translate_to_english: bool = True) -> str:
    """
    Processes audio bytes using Groq's ultra-fast whisper-large-v3 model.
    When translate_to_english=True (default), translates any spoken regional language 
    (Malayalam, Hindi, Tamil, etc.) directly into clean English text.
    For native English speech, it transcribes English verbatim.
    """
    if not audio_bytes:
        return ""

    client = AsyncGroq(api_key=GROQ_API_KEY)

    try:
        if translate_to_english:
            # Whisper translation endpoint translates speech from any language directly into English
            result = await client.audio.translations.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3",
                temperature=0.0,
            )
        else:
            result = await client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0,
            )

        transcribed_text = result.text.strip()
        print(f"🎙️ [Whisper Processed]: \"{transcribed_text}\"", flush=True)
        return transcribed_text

    except Exception as e:
        print(f"❌ Whisper Audio Error: {e}", flush=True)
        return ""


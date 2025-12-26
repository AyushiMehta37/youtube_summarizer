import os
import re
import json
import subprocess
import tempfile
import platform
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from gtts import gTTS

# ---------- Try to use faster-whisper if available ----------
try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
    print("⚡ Using faster-whisper for transcription.")
except ImportError:
    import whisper
    USE_FASTER_WHISPER = False
    print("🎧 Using standard whisper for transcription.")

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ---------- Helper: Parse and merge VTT subtitles ----------
def parse_vtt(vtt_path: str):
    raw = Path(vtt_path).read_text(encoding="utf-8")
    raw = re.sub(r"^(WEBVTT|Kind:.*|Language:.*)$", "", raw, flags=re.MULTILINE).strip()

    entries, current = [], None
    for line in map(str.strip, raw.splitlines()):
        if not line:
            continue

        match = re.match(r"(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)", line)
        if match:
            if current:
                entries.append(current)
            current = {"start": match.group(1), "end": match.group(2), "text": ""}
        elif current:
            clean = re.sub(r"<[^>]+>|align:start position:\d+%", "", line).strip()
            current["text"] += f" {clean}"

    if current:
        entries.append(current)

    for e in entries:
        e["text"] = re.sub(r"\s{2,}", " ", e["text"]).strip()

    return entries


def merge_and_clean(entries):
    merged = ""
    for entry in entries:
        line = entry["text"].strip()
        if not merged:
            merged = line
            continue

        merged_words, line_words = merged.split(), line.split()
        overlap = next(
            (i for i in range(min(len(merged_words), len(line_words)), 0, -1)
             if merged_words[-i:] == line_words[:i]), 0
        )
        merged += " " + " ".join(line_words[overlap:])
    return merged


# ---------- Global Whisper Initialization ----------
if USE_FASTER_WHISPER:
    whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
else:
    whisper_model = whisper.load_model("base")


def transcribe_audio(audio_path: str):
    """Transcribe an audio file using Whisper or Faster-Whisper."""
    print("🎙️ Transcribing audio...")

    if USE_FASTER_WHISPER:
        segments, info = whisper_model.transcribe(audio_path)
        text = " ".join([seg.text.strip() for seg in segments])
        print(f"🈶 Detected language: {info.language}")
    else:
        result = whisper_model.transcribe(audio_path, language=None)
        text = result["text"].strip()
        print(f"🈶 Detected language: {result.get('language', 'unknown')}")

    txt_path = Path(audio_path).with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print(f"📝 Transcript saved: {txt_path}")
    return text


# ---------- YouTube Caption or Audio ----------
def get_youtube_captions(url: str, output_dir="downloads"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        print("🟡 Checking for subtitles...")
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--skip-download", url],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        subs = data.get("subtitles") or data.get("automatic_captions") or {}

        if not subs:
            raise Exception("no_subtitles")

        lang = "en" if "en" in subs else next(iter(subs))
        subprocess.run(
            [
                "yt-dlp",
                "--sub-langs", lang,
                "--skip-download",
                "--write-auto-subs",
                "--sub-format", "vtt",
                "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
                url
            ],
            check=True, text=True
        )

        vtt_files = list(Path(output_dir).glob("*.vtt"))
        if not vtt_files:
            raise Exception("no_vtt_file")

        vtt_path = vtt_files[0]
        text = merge_and_clean(parse_vtt(vtt_path))
        txt_path = vtt_path.with_suffix(".txt")
        txt_path.write_text(text, encoding="utf-8")

        print(f"✅ Transcript saved from subtitles: {txt_path}")
        return {"type": "subtitles", "text_path": str(txt_path)}

    except Exception as e:
        print(f"🎧 No subtitles found ({e}) — using Whisper fallback.")

        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestaudio[ext=m4a]/bestaudio/best",
                "-x", "--audio-format", "mp3",
                "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
                url
            ],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print("❌ Audio download failed:")
            print(result.stderr)
            raise RuntimeError("yt-dlp audio download failed. Ensure ffmpeg is installed.")

        mp3_files = list(Path(output_dir).glob("*.mp3"))
        if not mp3_files:
            raise FileNotFoundError("No MP3 file found after download")

        mp3_path = mp3_files[0]
        text = transcribe_audio(mp3_path)
        return {"type": "audio_transcribed", "text": text}


# ---------- Summarization ----------
def summarize_transcript(text: str, model_name="gpt-4o-mini"):
    import requests
    import json
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert communicator summarizing YouTube videos for listeners. Create a smooth, conversational spoken summary focusing on clarity and flow. Avoid markdown or emojis. End with a one-line reflection."
            },
            {
                "role": "user",
                "content": text[:10000]
            }
        ],
        "temperature": 0.6,
    }

    print("⏳ Generating summary...")
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} — {response.text}")

    result = response.json()
    summary = result["choices"][0]["message"]["content"].strip()
    return summary



# ---------- Text-to-Speech ----------
def text_to_speech(text: str, output_path="summary_audio.mp3"):
    tts = gTTS(text)
    tts.save(output_path)
    print(f"🔊 Audio saved: {output_path}")

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", output_path])
        elif system == "Windows":
            os.startfile(output_path)
        else:
            subprocess.run(["xdg-open", output_path])
    except Exception as e:
        print(f"⚠️ Audio playback failed: {e}")


# ---------- Interactive Q&A ----------
def answer_questions(text: str, model_name="gpt-4o-mini"):
    print("\n💬 Entering Q&A Mode — type 'exit' to quit.\n")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.5,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL")
    )

    system_prompt = (
        "You are a helpful AI assistant answering questions about a YouTube video. "
        "Use only the transcript as context. If uncertain, say you don't have enough info. "
        "Be clear and natural, like explaining to a friend."
    )

    while True:
        question = input("❓ Your question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting Q&A mode.")
            break

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"Transcript:\n{text[:15000]}\n\nQuestion: {question}")
        ])

        response = (prompt | llm).invoke({}).content.strip()
        print(f"\n💡 {response}\n")

        try:
            tts = gTTS(response)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tts.save(tmp.name)
            if platform.system() == "Darwin":
                subprocess.run(["afplay", tmp.name])
            elif platform.system() == "Windows":
                os.startfile(tmp.name)
            else:
                subprocess.run(["xdg-open", tmp.name])
        except Exception:
            pass


# ---------- Main ----------
if __name__ == "__main__":
    load_dotenv()
    yt_url = input("Enter YouTube URL: ").strip()

    result = get_youtube_captions(yt_url)
    text = result.get("text") or Path(result["text_path"]).read_text(encoding="utf-8")

    if result["type"] in ["subtitles", "audio_transcribed"]:
        summary = summarize_transcript(text)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_txt = Path("downloads") / f"summary_{timestamp}.txt"
        out_txt.write_text(summary, encoding="utf-8")
        print(f"\n✅ Summary saved: {out_txt}\n\n📘 Preview:\n{summary[:800]}{'...' if len(summary) > 800 else ''}\n")

        out_audio = Path("downloads") / f"summary_{timestamp}.mp3"
        text_to_speech(summary, str(out_audio))

        answer_questions(text)

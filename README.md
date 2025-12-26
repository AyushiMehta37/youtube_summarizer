# YouTube Summarizer

`youtube_summarizer` is a Python tool that:

1. Downloads a YouTube video
2. Extracts subtitles (or transcribes audio if none exist)
3. Creates a natural, conversational AI summary
4. Saves the summary as:

   * a text file
   * an audio file
5. Supports interactive Q&A about the video

Useful for study, research, podcasts, revision notes, or saving time on long videos.

---

## Features

✔ Automatic subtitle detection (VTT)
✔ Whisper fallback for transcription
✔ OpenAI compatible LLMs for summaries
✔ Google Text to Speech audio output
✔ Interactive Q&A mode
✔ Local storage for transcripts and summaries
✔ CUDA acceleration when available

The core logic lives in `main.py`.

---

## How It Works

1. Fetch YouTube subtitles via `yt-dlp`
2. If none exist, download audio
3. Transcribe using:

   * `faster-whisper` (preferred), or
   * `openai-whisper`
4. Send the text to an OpenAI compatible API
5. Save results and generate speech
6. Enter Q&A mode

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AyushiMehta37/youtube_summarizer.git
cd youtube_summarizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install required system tools

You will need:

* ffmpeg
* yt-dlp

Windows (Chocolatey):

```bash
choco install ffmpeg
choco install yt-dlp
```

macOS (Homebrew):

```bash
brew install ffmpeg yt-dlp
```

Linux:

```bash
sudo apt install ffmpeg
pip install yt-dlp
```

---

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

Works with any OpenAI compatible endpoint.

---

## Usage

Run:

```bash
python main.py
```

When prompted, enter a YouTube URL.

The app will:

✔ fetch captions or audio
✔ generate a summary
✔ save files to `/downloads`
✔ play the audio summary
✔ start Q&A mode

Type `exit` to quit Q&A mode.

---

## Output Files

Stored in:

```
downloads/
```

| File Type         | Description                   |
| ----------------- | ----------------------------- |
| `.txt`            | AI generated summary          |
| `.mp3`            | Audio version of summary      |
| transcript `.txt` | Extracted or transcribed text |

---

## Example

```
Enter YouTube URL:
https://youtube.com/watch?v=example

Transcription started...
Summary in progress...
Saved: downloads/summary_20251226_153000.txt
Audio ready
Q&A Mode active
```

---

## Technology Stack

* Python
* yt-dlp
* Whisper / Faster Whisper
* LangChain
* OpenAI style APIs
* gTTS
* python-dotenv

---

## Troubleshooting

### yt-dlp audio download failed

Install ffmpeg and yt-dlp

### Whisper feels slow

Use a GPU enabled PyTorch environment

### No API key found

Check the `.env` file

### python3.11 not recognized (Windows)

```bash
python -m venv venv
```


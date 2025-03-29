# YouTube Knowledge Base

A local RAG-powered Q&A system for YouTube videos and playlists. Paste any
YouTube URL, the app transcribes it with Whisper, and you can chat with the
content using natural language.

---

## How It Works

1. Paste a YouTube video or playlist URL
2. `yt-dlp` downloads the audio
3. Whisper (local) transcribes it to text
4. LangChain splits the transcript into chunks
5. HuggingFace sentence-transformers embeds the chunks into a local FAISS index
6. When you ask a question, the top matching chunks are sent to the LLM
7. The LLM answers based only on the video content

---

## Tech Stack

| Component | Library |
|---|---|
| Audio download | yt-dlp |
| Transcription | OpenAI Whisper (local) |
| Chunking + RAG | LangChain |
| Vector store | FAISS (local) |
| Embeddings | HuggingFace sentence-transformers |
| LLM (local) | Ollama (llama3) |
| LLM (cloud) | OpenAI (gpt-4o-mini) |
| UI | Gradio |

---

## Installation

**Prerequisites:** Python 3.10+, ffmpeg in PATH

```bash
git clone https://github.com/your-username/yt-knowledge-base.git
cd yt-knowledge-base

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## LLM Backend

The backend is controlled by `LLM_BACKEND` in `.env`. Switch anytime and
restart the app — indexes are shared between both backends.

### Ollama (default, fully local)

```
# .env
LLM_BACKEND=ollama
```

Install Ollama from https://ollama.com/download, then pull the model:

```bash
ollama pull llama3
```

### OpenAI

```
# .env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-your-key-here
```

> Embeddings always run locally via HuggingFace — only the Q&A step uses
> the OpenAI API.

---

## Running the App

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## Usage

1. Paste a YouTube video or playlist URL
2. Click **Load Video** and wait for processing
3. Once status shows **Ready to chat**, ask questions in the chat box

Already-processed videos are cached — reloading the same URL is instant.

---

## Project Structure

```
yt-knowledge-base/
├── core/
│   ├── transcriber.py    # yt-dlp + Whisper
│   ├── chunker.py        # text splitting
│   ├── embedder.py       # FAISS index
│   ├── retriever.py      # similarity search
│   └── llm.py            # Ollama / OpenAI
├── transcripts/          # cached audio + transcripts
├── indexes/              # FAISS indexes
├── pipeline.py           # orchestration
├── app.py                # Gradio UI
├── config.py             # all settings
└── .env                  # local config (not committed)
```

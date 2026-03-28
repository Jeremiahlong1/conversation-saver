# Claude Conversation Saver

Hit Claude's chat limit? Save your full conversation and generate a **context primer** — a structured summary you paste at the top of a new Claude chat to resume exactly where you left off.

## What it does

- **Fetches** any public Claude share link server-side (no CORS issues)
- **Extracts** the full conversation word for word
- **Generates** an AI-powered context primer using Claude API
- **Downloads** transcript as .txt or .md

## Deploy to Render (free, 3 steps)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect repo
3. Add your `ANTHROPIC_API_KEY` in the Render environment variables dashboard

Live in ~2 minutes.

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py
# open http://localhost:5050
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | For context primer generation |
| `PORT` | No | Defaults to 5050 locally |

## How the parsing works

Three strategies in order of reliability:
1. `__NEXT_DATA__` JSON blob embedded in the share page (most reliable)
2. Script tag scanning for embedded JSON
3. Regex pattern matching for role/content objects

If all three fail, use the HTML paste fallback — save the page in your browser (Ctrl+S → Webpage, Complete) and paste the source.

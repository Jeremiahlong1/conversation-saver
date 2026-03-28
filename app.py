"""
Claude Conversation Saver
-------------------------
Fetches a public Claude share link server-side,
extracts the full conversation, and generates an
AI-powered context primer so users can resume
in a new chat without losing anything.

Deploy: Render.com (free tier)
Run locally: python app.py
"""

import json
import os
import re
from urllib.parse import urlparse

import requests
from anthropic import Anthropic
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client  = Anthropic(api_key=ANTHROPIC_API_KEY)

FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Parsing ────────────────────────────────────────────────────────────────

def find_messages_array(obj, depth=0):
    """Walk any JSON structure to find the messages array."""
    if depth > 15 or obj is None:
        return None
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and obj[0].get("role") in ("human", "assistant"):
            return obj
        for item in obj:
            r = find_messages_array(item, depth + 1)
            if r:
                return r
    elif isinstance(obj, dict):
        for key in ("messages", "chat_messages", "conversation", "turns", "items"):
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("role"):
                return v
        for v in obj.values():
            r = find_messages_array(v, depth + 1)
            if r:
                return r
    return None


def normalize_message(msg, idx):
    role = (msg.get("role") or msg.get("sender") or "").lower()
    is_human = "human" in role or role == "user"

    content = ""
    raw = msg.get("content")
    if isinstance(raw, str):
        content = raw
    elif isinstance(raw, list):
        content = "\n".join(
            b.get("text", "") for b in raw
            if isinstance(b, dict) and b.get("type") == "text"
        )
    elif msg.get("text"):
        content = str(msg["text"])

    content = content.strip()
    if not content:
        return None

    return {
        "role": "human" if is_human else "assistant",
        "content": content,
        "turn": idx + 1,
    }


def parse_html(html):
    # Strategy 1: __NEXT_DATA__
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">([\s\S]*?)</script>',
        html
    )
    if m:
        try:
            data = json.loads(m.group(1))
            raw = find_messages_array(data)
            if raw:
                messages = [normalize_message(r, i) for i, r in enumerate(raw)]
                messages = [x for x in messages if x]
                if messages:
                    return {"messages": messages, "source": "next_data"}
        except Exception:
            pass

    # Strategy 2: scan script tags
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        text = script.string or ""
        if '"human"' not in text and '"assistant"' not in text:
            continue
        for jm in re.finditer(r'\{[\s\S]{100,}\}', text):
            try:
                obj = json.loads(jm.group(0))
                raw = find_messages_array(obj)
                if raw:
                    messages = [normalize_message(r, i) for i, r in enumerate(raw)]
                    messages = [x for x in messages if x]
                    if messages:
                        return {"messages": messages, "source": "script_scan"}
            except Exception:
                continue

    # Strategy 3: regex blob scan
    raw_found = []
    for hit in re.finditer(
        r'\{"role"\s*:\s*"(?:human|assistant)"[\s\S]{1,8000}?\}(?=\s*[,\]\}])',
        html
    ):
        try:
            obj = json.loads(hit.group(0))
            if obj.get("role") and obj.get("content"):
                raw_found.append(obj)
        except Exception:
            continue

    if raw_found:
        messages = [normalize_message(r, i) for i, r in enumerate(raw_found)]
        messages = [x for x in messages if x]
        if messages:
            return {"messages": messages, "source": "regex_scan"}

    return {
        "messages": [],
        "source": "none",
        "error": (
            "Could not find conversation data in this page. "
            "Try saving the full page in your browser (Ctrl+S → Webpage Complete) "
            "and using the HTML paste option."
        ),
    }


# ── Context Primer Generation ──────────────────────────────────────────────

PRIMER_SYSTEM = """You are an expert at analyzing AI conversations and creating concise, 
structured context documents that allow someone to seamlessly resume a conversation in a 
new chat session. Your output will be pasted at the start of a new Claude conversation."""

PRIMER_PROMPT = """Analyze this Claude conversation and create a structured context primer 
that captures everything needed to resume it perfectly in a new chat.

Format your response exactly like this:

## CONVERSATION CONTEXT PRIMER
*Paste this at the start of your new Claude chat*

---

**GOAL**
[One sentence: what the user was trying to accomplish]

**BACKGROUND**
[2-3 sentences of essential context Claude needs to understand the situation]

**WHAT WAS DECIDED / BUILT**
[Bullet points of key decisions, conclusions, or things that were created]

**CURRENT STATE**
[Where things stood at the end — what's done, what's in progress, what's next]

**IMPORTANT DETAILS**
[Any specific constraints, preferences, requirements, or nuances Claude needs to know]

**FILES & LINKS REFERENCED**
[List any files, URLs, or resources mentioned — name and brief description]

**OPEN QUESTIONS**
[Anything unresolved or next steps the user was heading toward]

---
*Resume from here. The user will continue where they left off.*

---

Here is the full conversation:

{conversation}"""


def build_conversation_text(messages):
    parts = []
    for msg in messages:
        role = "User" if msg["role"] == "human" else "Claude"
        parts.append(f"[{role}]\n{msg['content']}")
    return "\n\n---\n\n".join(parts)


def generate_primer(messages):
    """Call Claude API to generate the context primer."""
    if not ANTHROPIC_API_KEY:
        return None, "ANTHROPIC_API_KEY not configured on server."

    conversation_text = build_conversation_text(messages)

    # Truncate if extremely long to stay within token limits
    # Rough estimate: 1 token ≈ 4 chars. Keep under ~150k tokens input.
    max_chars = 500_000
    if len(conversation_text) > max_chars:
        conversation_text = conversation_text[:max_chars] + "\n\n[Conversation truncated for length]"

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=PRIMER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": PRIMER_PROMPT.format(conversation=conversation_text),
                }
            ],
        )
        primer_text = response.content[0].text
        return primer_text, None
    except Exception as e:
        return None, str(e)


# ── Text builders ──────────────────────────────────────────────────────────

def build_txt(messages):
    sep = "\n\n" + "─" * 60 + "\n\n"
    parts = []
    for i, msg in enumerate(messages):
        role = "YOU" if msg["role"] == "human" else "CLAUDE"
        parts.append(f"[{role} — Turn {i + 1}]\n{msg['content']}")
    return sep.join(parts)


def build_md(messages):
    parts = []
    for i, msg in enumerate(messages):
        role = "**You**" if msg["role"] == "human" else "**Claude**"
        parts.append(f"### {role} *(Turn {i + 1})*\n\n{msg['content']}")
    return "\n\n---\n\n".join(parts)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/fetch", methods=["POST"])
def api_fetch():
    body = request.get_json(force=True, silent=True) or {}
    url  = (body.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    parsed = urlparse(url)
    if "claude.ai" not in (parsed.netloc or ""):
        return jsonify({"error": "Must be a claude.ai share link"}), 400

    try:
        resp = requests.get(url, headers=FETCH_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 404:
            return jsonify({"error": "Share link not found. It may be private or deleted."}), 404
        return jsonify({"error": f"claude.ai returned HTTP {code}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Try again."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 502

    result = parse_html(resp.text)

    if not result["messages"]:
        return jsonify({"error": result.get("error", "No messages found.")}), 422

    messages = result["messages"]

    return jsonify({
        "messages": messages,
        "source": result["source"],
        "stats": {
            "message_count": len(messages),
            "turn_count": len([m for m in messages if m["role"] == "human"]),
            "char_count": sum(len(m["content"]) for m in messages),
        },
        "transcript_txt": build_txt(messages),
        "transcript_md": build_md(messages),
    })


@app.route("/api/parse", methods=["POST"])
def api_parse():
    body = request.get_json(force=True, silent=True) or {}
    html = (body.get("html") or "").strip()

    if not html:
        return jsonify({"error": "No HTML provided"}), 400

    result = parse_html(html)

    if not result["messages"]:
        return jsonify({"error": result.get("error", "No messages found.")}), 422

    messages = result["messages"]

    return jsonify({
        "messages": messages,
        "source": result["source"],
        "stats": {
            "message_count": len(messages),
            "turn_count": len([m for m in messages if m["role"] == "human"]),
            "char_count": sum(len(m["content"]) for m in messages),
        },
        "transcript_txt": build_txt(messages),
        "transcript_md": build_md(messages),
    })


@app.route("/api/primer", methods=["POST"])
def api_primer():
    """Generate AI context primer from already-parsed messages."""
    body     = request.get_json(force=True, silent=True) or {}
    messages = body.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    primer, error = generate_primer(messages)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({"primer": primer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  Claude Conversation Saver")
    print(f"  Running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

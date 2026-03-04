# I am writing this in a python file because all this is suggested by Claude and the scope for mistake is low.

import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

OLLAMA_BASE_URL  = "http://localhost:11434/v1"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings" # It's this way because Chromabd's embedding function needs this endpoint to work, even though it's not the same as the base URL for the LLM.
MODEL_NAME       = "llama3.2"
CHROMA_PATH      = "./stamper_memory_db"
COLLECTION_NAME  = "stamper_sessions"

_llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key="OLLAMA_API_KEY")

def _llm_call(prompt:str):
    response = _llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content.strip()

def _get_collections():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.OllamaEmbeddingFunction(url=OLLAMA_EMBED_URL, model_name=MODEL_NAME)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn, metadata={"hnsw:space":"cosine"})


_session: dict = {}

def start_session(name, hobbies):
    global _session
    session_id = str(uuid.uuid4())
    temp_path = Path(tempfile.gettempdir()) / f"stamper_{session_id}.tmp"
    
    
    _session = {
        "id":         session_id,
        "name":       name,
        "hobbies":    hobbies,
        "turns":      [],
        "started_at": datetime.now().isoformat(),
        "temp_path":  temp_path,
    }
    _write_temp()
    print(f"Stamper memory session started with ID: {temp_path.name}")
    return session_id
    
def log(role, content):
    if not _session:
        raise RuntimeError("Call start_session() before log()")
    _session['turns'].append({
        "turn": len(_session["turns"])+1,
        "role": role,
        "content":content,
        "timestamp":datetime.now().isoformat()
        
    })
    _write_temp()
    
    
def retrieve_context(query, n:int = 2):
    col = _get_collections()
    count = col.count()
    if count==0:
        return ""
    
    results = col.query(
        query_texts=[query],
        n_results=min(n, count), 
        include=[ "documents", "metadatas", "distances"]
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]
    
    if not docs:
        return ""
    
    snippets = []
    for doc, meta, dist in zip(docs, metas, distances):
        relevance = round((1-dist)*100, 1)
        snippets.append(f"[{meta.get('date','?')} | {relevance}% match]\n{doc}")
        
    return (
        "Memories from past conversations with this user"
        "(weave in naturally, do not quote verbatim):\n\n"
        + "\n\n---\n\n".join(snippets)
        + "\n"
    )

def get_recent_turns(n: int = 10):
    if not _session:
        return []
    return [
        {"role": t["role"], "content": t["content"]}
        for t in _session["turns"][-n:]
    ]
def end_session() -> dict:
    if not _session:
        raise RuntimeError("No active session to end.")

    print("[Stamper Memory] Ending session — summarising…")
    _write_temp()

    structured = _structure_with_llm()
    _store_in_chroma(structured)
    _cleanup_temp()

    print("[Stamper Memory] ✓ Saved to ChromaDB.")
    return structured


def _write_temp() -> None:
    data = {
        "session_id": _session["id"],
        "name":       _session["name"],
        "hobbies":    _session["hobbies"],
        "started_at": _session["started_at"],
        "ended_at":   datetime.now().isoformat(),
        "temp_path":  str(_session["temp_path"]),
        "turns":      _session["turns"],
    }
    _session["temp_path"].write_text(json.dumps(data, indent=2), encoding="utf-8")


def _structure_with_llm() -> dict:
    transcript = "\n".join(
        f"{t['role'].upper()}: {t['content']}"
        for t in _session["turns"]
    )
    prompt = f"""Analyse this conversation and return ONLY a valid JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{{
  "summary":   "<2-3 sentences on what was discussed>",
  "topics":    ["<topic1>", "<topic2>"],
  "key_facts": ["<important preference or fact the user revealed>"],
  "mood":      "<overall mood: positive | neutral | negative | mixed>",
  "language":  "<primary language used>"
}}

User name: {_session['name']}
User hobbies: {_session['hobbies']}

TRANSCRIPT:
{transcript}
"""
    raw = _llm_call(prompt)

    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]

    try:
        structured = json.loads(raw.strip())
    except json.JSONDecodeError:
        structured = {
            "summary":   raw[:400],
            "topics":    [],
            "key_facts": [],
            "mood":      "neutral",
            "language":  "en",
        }

    out_dir = Path(CHROMA_PATH) / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{_session['id']}.json"
    out_file.write_text(
        json.dumps({
            "meta": structured,
            "transcript": {
                "session_id": _session["id"],
                "name":       _session["name"],
                "hobbies":    _session["hobbies"],
                "started_at": _session["started_at"],
                "turns":      _session["turns"],
            }
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[Stamper Memory] JSON  →  {out_file.name}")
    return structured


def _store_in_chroma(structured: dict) -> None:
    col        = _get_collection()
    session_id = _session["id"]
    date_str   = _session["started_at"][:10]

    summary_doc = structured.get("summary", "No summary.")

    facts_doc = (
        f"User: {_session['name']} | Hobbies: {_session['hobbies']}\n"
        f"Topics: {', '.join(structured.get('topics', []))}\n"
        + "\n".join(f"• {f}" for f in structured.get("key_facts", []))
    )

    meta = {
        "session_id": session_id,
        "date":       date_str,
        "name":       _session["name"],
        "hobbies":    _session["hobbies"],
        "mood":       structured.get("mood", "neutral"),
        "topics":     ", ".join(structured.get("topics", [])),
        "turns":      str(len(_session["turns"])),
    }

    col.upsert(
        ids       =[f"{session_id}_summary", f"{session_id}_facts"],
        documents =[summary_doc, facts_doc],
        metadatas =[meta, meta],
    )


def _cleanup_temp() -> None:
    p = _session.get("temp_path")
    if p and Path(p).exists():
        Path(p).unlink()
        print("[Stamper Memory] Temp file removed.")
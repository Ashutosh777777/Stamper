"""
scraper.py  —  Stamper Edition v2
===================================
Fixes:
  - Replaces fragile DuckDuckGo HTML scraping with the duckduckgo-search library
  - Wikipedia is now a fallback AFTER web search, not the primary route
  - LLM is explicitly told not to add anything beyond what the search results say
  - web_context is scoped per-query so stale results never bleed into the next turn

Install requirement:
  pip install duckduckgo-search

Public API:
  should_search(query)   → bool  : does this query need the web?
  search(query)          → str   : fetch real results and return a grounded answer
  summarise_url(url)     → str   : summarise a specific URL
"""

import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from openai import OpenAI


# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME      = "llama3.2"
MAX_CHARS       = 3000

_llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key="OLLAMA_API_KEY")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}


# ── Public API ─────────────────────────────────────────────────────────────────

def should_search(query: str) -> bool:
    """
    Asks the LLM whether this query needs a web search.
    Specifically looks for: recent events, specific people/films/products,
    anything that could have changed or that the LLM might not know.
    """
    prompt = (
        f"You are deciding whether a question needs a live web search to answer accurately.\n"
        f"Answer YES if the query is about: a specific person, film, product, recent event, "
        f"news, or anything that might not be in your training data.\n"
        f"Answer NO only if it is a general knowledge question, casual conversation, "
        f"or something you are completely certain about.\n\n"
        f"Query: {query}\n\n"
        f"Reply with only YES or NO."
    )
    result = _llm_call(prompt).strip().upper()
    return result.startswith("YES")


def search(query: str) -> str:
    """
    Routes the query to the right method and returns a grounded answer.
    Never lets the LLM add information beyond what was actually found.
    """
    # Route 1: direct URL in the query
    url = _extract_url(query)
    if url:
        return summarise_url(url)

    # Route 2: DuckDuckGo web search (primary for everything)
    result = _search_web(query)
    if result:
        return result

    # Route 3: Wikipedia fallback (only if web search found nothing)
    result = _search_wikipedia(query)
    if result:
        return result

    return "I searched the web but couldn't find reliable information on that."


def summarise_url(url: str) -> str:
    """Fetches a URL and returns a grounded LLM summary."""
    raw = _fetch_page(url)
    if not raw:
        return f"I couldn't access {url}."

    prompt = (
        f"Here is the content of a webpage. "
        f"Summarise ONLY what is written here in 3-5 sentences. "
        f"Do not add anything that is not in the text below.\n\n"
        f"{raw}"
    )
    return _llm_call(prompt)


# ── DuckDuckGo search (primary) ────────────────────────────────────────────────

def _search_web(query: str) -> str:
    """
    Uses the duckduckgo-search library to get real search results.
    Much more reliable than scraping the HTML search page.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return ""

        # Build a grounded context block from the actual results
        context = ""
        for r in results:
            title   = r.get("title", "")
            snippet = r.get("body", "")
            source  = r.get("href", "")
            context += f"Source: {source}\n{title}\n{snippet}\n\n"

        context = context[:MAX_CHARS]

    except Exception as e:
        return ""

    # Strict prompt — LLM must only use what's in the results
    prompt = (
        f"Answer the following question using ONLY the search results provided below. "
        f"If the search results do not contain enough information to answer confidently, "
        f"say exactly: 'I found some results but they don't clearly answer that.' "
        f"Do not add any information from your own knowledge. "
        f"Answer in 2-4 sentences, conversationally, no bullet points, no markdown.\n\n"
        f"Question: {query}\n\n"
        f"Search results:\n{context}"
    )
    return _llm_call(prompt)


# ── Wikipedia fallback ─────────────────────────────────────────────────────────

def _search_wikipedia(query: str) -> str:
    """
    Wikipedia API lookup — used only as a fallback when DuckDuckGo finds nothing.
    """
    api_url = "https://en.wikipedia.org/w/api.php"

    # Search for the best matching article
    try:
        resp = requests.get(api_url, params={
            "action": "query", "list": "search",
            "srsearch": query, "format": "json", "srlimit": 1,
        }, headers=HEADERS, timeout=8)
        results = resp.json()["query"]["search"]
        if not results:
            return ""
        page_title = results[0]["title"]
    except Exception:
        return ""

    # Fetch the article intro
    try:
        resp = requests.get(api_url, params={
            "action": "query", "titles": page_title,
            "prop": "extracts", "exintro": True,
            "explaintext": True, "format": "json",
        }, headers=HEADERS, timeout=8)
        pages   = resp.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
        if not extract:
            return ""
    except Exception:
        return ""

    prompt = (
        f"Answer the following question using ONLY the Wikipedia text below. "
        f"If the text does not answer the question, say: "
        f"'I couldn't find clear information on that.' "
        f"Answer in 2-4 sentences, conversationally, no bullet points.\n\n"
        f"Question: {query}\n\n"
        f"Wikipedia:\n{extract[:MAX_CHARS]}"
    )
    return _llm_call(prompt)


# ── Page fetcher ───────────────────────────────────────────────────────────────

def _fetch_page(url: str) -> str:
    """Fetches a URL and returns clean body text."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup     = BeautifulSoup(response.content, "html.parser")
        title    = soup.title.string if soup.title else ""
        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()
            text = soup.body.get_text(separator="\n", strip=True)
        else:
            text = ""
        return (title + "\n\n" + text)[:MAX_CHARS]
    except Exception:
        return ""


# ── URL extractor ──────────────────────────────────────────────────────────────

def _extract_url(text: str):
    match = re.search(r'https?://[^\s]+', text)
    return match.group(0) if match else None


# ── LLM call ───────────────────────────────────────────────────────────────────

def _llm_call(prompt: str) -> str:
    resp = _llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()
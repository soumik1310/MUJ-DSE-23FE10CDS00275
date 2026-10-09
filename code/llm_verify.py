import os
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

from prompts import build_system_prompt, build_user_prompt, parse_llm_json

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_SOURCES = 8


def fallback(message):
    return {"verdict": "Unverified", "confidence": 0.0, "reasoning": message,
            "claims": [], "sources": [], "grounded": False}


def resolve(url):
    try:
        with requests.get(url, allow_redirects=True, timeout=5, stream=True) as r:
            return r.url
    except requests.RequestException:
        return url


def extract_sources(response):
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks or []
    except (AttributeError, IndexError, TypeError):
        return []

    found = [(c.web.title, c.web.uri) for c in chunks if getattr(c, "web", None) and c.web.uri]
    found = found[:MAX_SOURCES]
    if not found:
        return []
    with ThreadPoolExecutor(max_workers=8) as pool:
        urls = list(pool.map(lambda t: resolve(t[1]), found))

    sources, seen = [], set()
    for (title, _), url in zip(found, urls):
        if url not in seen:
            seen.add(url)
            sources.append({"title": title or url, "url": url})
    return sources


def verify_with_gemini(text):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=build_user_prompt(text),
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        result = parse_llm_json(response.text or "")
        result["sources"] = extract_sources(response)
        result["grounded"] = bool(result["sources"])
        return result
    except Exception as e:
        return fallback(f"LLM call failed ({type(e).__name__}). Check your API key and model name.")

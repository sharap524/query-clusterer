#!/usr/bin/env python3
"""
Claude-based translator with persistent cache and batch processing.

Why Claude instead of DeepL:
- DeepL isn't available in all regions (Armenia, Russia, Belarus, etc.)
- Claude handles SEO-specific context better (brand names, abbreviations,
  intent keywords) — it knows "купить iphone 15 pro" is a shopping query,
  not literary Russian.
- Haiku 4.5 is extremely cheap for this workload:
  ~$0.0002 per short query at standard rates; with prompt caching
  on re-runs, effectively free.
- One API call translates many queries at once (batch inside the prompt),
  which is faster than DeepL's per-request limit.

Cost check (Haiku 4.5, April 2026):
  $1 per 1M input tokens, $5 per 1M output tokens.
  A short SEO query ≈ 10-30 tokens in + 5-15 tokens out.
  10 000 unique queries ≈ $0.50-$2.00 total, one-time.
  Re-runs hit the cache: ~$0.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable

from anthropic import Anthropic, APIError, APIStatusError


# Cheapest current-generation Claude model — more than enough for translation.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# How many queries to translate per API call. Claude handles hundreds easily,
# but we cap at 50 so a single transient failure doesn't waste a huge batch.
BATCH_SIZE = 50

# Safety cap per single query — SEO keywords are always short.
MAX_SINGLE_TEXT_CHARS = 2000

# ISO-639-1 codes Claude may report. Anything else gets mapped to 'und'.
KNOWN_LANGS = {
    "en", "ru", "de", "fr", "es", "it", "pt", "pl", "uk", "nl", "tr",
    "ja", "ko", "zh", "ar", "hi", "th", "vi", "id", "cs", "sv", "da",
    "no", "nb", "fi", "el", "he", "ro", "hu", "bg", "sk", "hr", "sl",
    "lt", "lv", "et", "ka", "hy", "az", "kk", "uz", "sr",
}


SYSTEM_PROMPT = """You are an SEO query translator. For each input query, you will:
1. Detect the source language (use ISO 639-1 code, lowercase, e.g. "ru", "en", "de")
2. Translate to natural English as a native speaker would search for the same thing
3. Preserve brand names, model numbers, and technical terms exactly as-is

CRITICAL rules:
- Keep the translation SHORT and SEARCH-LIKE, not a full sentence. "купить iphone 15" → "buy iphone 15", NOT "I would like to purchase an iPhone 15".
- Do not add context the user didn't ask for
- Brands stay in original form: "Самсунг" → "Samsung", not "Samsung Corporation"
- If the query is already in English, still output it (normalized/cleaned) and set language to "en"
- If you cannot identify the language, use "und"

Output format: A single JSON array, one object per input, in the SAME ORDER as inputs. No prose, no markdown fences, just the JSON array.

Each object MUST have exactly these two fields:
{"lang": "ru", "en": "buy iphone 15 pro"}

Example input:
1. купить iphone 15 pro
2. best pizza near me
3. preis samsung galaxy s24

Example output:
[{"lang":"ru","en":"buy iphone 15 pro"},{"lang":"en","en":"best pizza near me"},{"lang":"de","en":"samsung galaxy s24 price"}]"""


@dataclass
class TranslationResult:
    text: str              # translated English text
    detected_lang: str     # lowercase ISO code, or 'und'
    source: str            # 'claude' | 'cache' | 'passthrough' | 'error'


class ClaudeTranslator:
    """
    Anthropic Claude client with SQLite cache.

    Usage:
        tr = ClaudeTranslator(api_key=os.environ["ANTHROPIC_API_KEY"],
                              cache_db="translations.db")
        results = tr.translate_batch(["купить iphone", "buy iphone"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_db: str = "translations.db",
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        self.client = Anthropic(api_key=key, timeout=timeout)
        self.model = model
        self.cache_db = cache_db
        self._init_cache()

    # ------------------------------------------------------------------ cache

    def _init_cache(self):
        conn = sqlite3.connect(self.cache_db)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS translation_cache (
                source_text TEXT PRIMARY KEY,
                translated_text TEXT NOT NULL,
                detected_lang TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_detected_lang ON translation_cache(detected_lang)")
        conn.commit()
        conn.close()

    def _cache_get_many(self, texts: list[str]) -> dict[str, TranslationResult]:
        if not texts:
            return {}
        conn = sqlite3.connect(self.cache_db)
        cur = conn.cursor()
        placeholders = ",".join("?" * len(texts))
        cur.execute(
            f"SELECT source_text, translated_text, detected_lang "
            f"FROM translation_cache WHERE source_text IN ({placeholders})",
            texts,
        )
        rows = cur.fetchall()
        conn.close()
        return {
            src: TranslationResult(text=tr, detected_lang=lang, source="cache")
            for src, tr, lang in rows
        }

    def _cache_put_many(self, pairs: list[tuple[str, TranslationResult]]):
        if not pairs:
            return
        conn = sqlite3.connect(self.cache_db)
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO translation_cache "
            "(source_text, translated_text, detected_lang, model) "
            "VALUES (?, ?, ?, ?)",
            [(src, r.text, r.detected_lang, self.model) for src, r in pairs],
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ api

    def _call_claude(self, texts: list[str]) -> list[tuple[str, str]]:
        """
        Translate one batch via Claude. Returns list of (translated, lang)
        in input order.
        """
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        user_msg = f"Translate these {len(texts)} queries:\n\n{numbered}"

        last_err = None
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=min(8000, len(texts) * 80 + 200),
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            # Cache the system prompt — identical across all batches,
                            # so we pay the ~400-token write once and 10% on reads.
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                ).strip()

                parsed = self._parse_json_array(raw, expected=len(texts))
                return parsed
            except (APIStatusError, APIError) as e:
                status = getattr(e, "status_code", None)
                if status == 429 or (status and status >= 500):
                    time.sleep(1.5 ** attempt)
                    last_err = e
                    continue
                raise
            except ValueError as e:
                # JSON parsing failure — retry once
                last_err = e
                if attempt < 2:
                    time.sleep(1.0)
                    continue
                raise

        raise last_err or RuntimeError("Claude call failed without specific error")

    @staticmethod
    def _parse_json_array(raw: str, expected: int) -> list[tuple[str, str]]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            m = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if not m:
                raise ValueError(f"Claude returned non-JSON: {raw[:200]!r}") from e
            data = json.loads(m.group(0))

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        if len(data) != expected:
            raise ValueError(f"Expected {expected} items, got {len(data)}")

        out: list[tuple[str, str]] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not an object: {item!r}")
            lang = str(item.get("lang", "und")).lower().strip()
            text = str(item.get("en", "")).strip()
            if not text:
                raise ValueError(f"Item {i} has empty 'en' field")
            if lang not in KNOWN_LANGS:
                lang = "und"
            out.append((text, lang))
        return out

    # ------------------------------------------------------------------ public

    def translate_batch(
        self,
        texts: Iterable[str],
        use_cache: bool = True,
        progress: bool = True,
    ) -> list[TranslationResult]:
        """
        Translate a list of texts. Returns results in input order.
        Deduplicates, uses cache, batches uncached via Claude, and falls back
        to passthrough on errors so the pipeline never crashes.
        """
        texts = list(texts)
        if not texts:
            return []

        unique: list[str] = []
        seen: set[str] = set()
        for t in texts:
            if t not in seen and len(t) <= MAX_SINGLE_TEXT_CHARS:
                unique.append(t)
                seen.add(t)

        cache_hits: dict[str, TranslationResult] = {}
        if use_cache:
            cache_hits = self._cache_get_many(unique)

        misses = [t for t in unique if t not in cache_hits]
        fresh: dict[str, TranslationResult] = {}

        if misses:
            if progress:
                print(f"🤖 Claude: {len(misses)} new texts, {len(cache_hits)} cached")

            for i in range(0, len(misses), BATCH_SIZE):
                batch = misses[i : i + BATCH_SIZE]
                try:
                    api_out = self._call_claude(batch)
                    for src, (translated, lang) in zip(batch, api_out):
                        fresh[src] = TranslationResult(
                            text=translated, detected_lang=lang, source="claude"
                        )
                    if progress:
                        done = min(i + BATCH_SIZE, len(misses))
                        print(f"   batch {done}/{len(misses)} ✓")
                except Exception as e:
                    print(f"⚠️  Claude batch failed ({type(e).__name__}: {e}). "
                          f"Falling back to passthrough for {len(batch)} items.")
                    for src in batch:
                        fresh[src] = TranslationResult(
                            text=src, detected_lang="und", source="error"
                        )

            to_cache = [(src, r) for src, r in fresh.items() if r.source == "claude"]
            self._cache_put_many(to_cache)

        lookup = {**cache_hits, **fresh}
        out: list[TranslationResult] = []
        for t in texts:
            if t in lookup:
                out.append(lookup[t])
            else:
                out.append(TranslationResult(text=t, detected_lang="und", source="passthrough"))
        return out

    def estimate_cost(self, num_queries: int, avg_chars_per_query: int = 30) -> dict:
        """
        Rough cost estimate for a fresh run (no cache hits).
        Uses Haiku 4.5 rates: $1/Mtok input, $5/Mtok output.
        """
        input_tokens_per_query = max(10, avg_chars_per_query // 2 + 5)
        output_tokens_per_query = max(8, avg_chars_per_query // 3 + 3)

        system_tokens = 400  # system prompt, cached after first call

        total_input = num_queries * input_tokens_per_query + system_tokens
        total_output = num_queries * output_tokens_per_query

        cost = total_input * 1.0 / 1_000_000 + total_output * 5.0 / 1_000_000
        return {
            "num_queries": num_queries,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(cost, 4),
            "note": "Haiku 4.5 rates, uncached. Cache hits cost ~10% of input.",
        }

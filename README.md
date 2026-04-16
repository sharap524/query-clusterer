# 🔍 SEO Query Clusterer v2

Professional semantic clustering for SEO queries — powered by **Claude API** for high-quality translation and language detection.

## What changed in v2

| | v1 | v2 |
|---|---|---|
| Translation | Dictionary of ~30 words | **Claude Haiku 4.5** — real LLM-quality translation |
| Language detection | `langid` (often wrong on short queries) | **Claude** (context-aware, handles brand names) |
| Cluster names | Mix of original languages | Clean English, built from translations |
| Re-running | Re-translates every time | **SQLite cache** — zero waste |
| API cost | N/A | **~$1–2 per 10 000 queries**, then ~$0 via cache |
| Regional availability | N/A | Works in Armenia, Georgia, and most countries |

---

## Setup

### 1. Get an Anthropic API key

Sign up at [console.anthropic.com](https://console.anthropic.com). You'll need to add **$5 minimum** to your balance to start using the API. Armenia is supported.

Create a key: Settings → API Keys → Create Key. Copy it (starts with `sk-ant-...`).

### 2. Add the key to GitHub secrets

Repo → Settings → Secrets and variables → Actions → New repository secret:
- Name: `ANTHROPIC_API_KEY`
- Value: your key

### 3. Upload files to repo

- `clusterer.py`
- `translator.py`
- `requirements.txt`
- `queries.txt`
- `.github/workflows/cluster.yml` (rename `cluster.yml` and put it in this path)

### 4. Run

Actions → SEO Query Clustering → Run workflow → set threshold → Run

---

## Pipeline

```
load → translate (Claude) → cluster (on English text) → export
```

Each step is independent and idempotent. Re-clustering at a new threshold does **not** re-call Claude — translations are cached.

### Full run in one command

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python clusterer.py all --input queries.txt --threshold 0.75
```

### Step by step

```bash
# 1. Load queries into DB
python clusterer.py load --input queries.txt

# 2. Translate (uses cache on repeat runs)
python clusterer.py translate

# 3. Cluster on English translations
python clusterer.py cluster --threshold 0.75

# 4. Export
python clusterer.py export --output results.json
python clusterer.py export --output results.csv

# 5. Stats
python clusterer.py stats
```

### Tweaking threshold without re-translating

```bash
python clusterer.py cluster --threshold 0.80  # stricter
python clusterer.py cluster --threshold 0.70  # looser
```

Translations stay in `seo_queries.db` and Claude cache in `translations.db` — this is free.

---

## Output

### JSON
```json
{
  "clusters": {
    "Iphone 15 Pro Price": {
      "query_count": 7,
      "languages": {"ru": 4, "en": 3},
      "centroid_query": "iphone 15 pro price",
      "centroid_query_en": "iphone 15 pro price",
      "queries": [
        {
          "query": "купить iphone 15 pro",
          "language": "ru",
          "language_name": "Russian",
          "translation_en": "buy iphone 15 pro"
        }
      ]
    }
  }
}
```

### CSV columns
`Cluster | Cluster ID | Query | Language | Language Name | Translation (EN) | Translation Source`

---

## Cost

Claude Haiku 4.5: **$1 per 1M input tokens, $5 per 1M output tokens**.

Typical SEO query ≈ 10–30 tokens in, 5–15 out. Real-world costs:

| Queries | First run | Re-runs (cached) |
|---|---|---|
| 1 000 | ~$0.10 | $0 |
| 10 000 | ~$1.00 | $0 |
| 100 000 | ~$10 | $0 |

The free-tier trial credit ($5 on signup) covers ~50 000 queries.

Check your current usage at [console.anthropic.com/settings/usage](https://console.anthropic.com/settings/usage).

### Estimate cost programmatically

```python
from translator import ClaudeTranslator
t = ClaudeTranslator()
print(t.estimate_cost(num_queries=10000, avg_chars_per_query=30))
# {'num_queries': 10000, 'estimated_cost_usd': 0.85, ...}
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--threshold` | 0.75 | Cosine-similarity cutoff. Higher = stricter. |
| `--input` | queries.txt | Input file (TXT or CSV) |
| `--column` | auto-detect | CSV column with queries |
| `--anthropic-key` | from env | Claude API key |
| `--force-retranslate` | off | Re-translate everything, ignoring cache |
| `--no-translations` | off | Cluster on raw queries (old v1 behaviour) |

### Threshold guide
- `0.65` — loose, broader topics per cluster
- `0.75` — balanced (recommended)
- `0.85` — strict, tight groups
- `0.90` — near-duplicates only

---

## Databases

- `seo_queries.db` — main store (queries, translations, cluster assignments)
- `translations.db` — Claude cache (keyed by source text)

Both are SQLite. The workflow caches `translations.db` between runs so you don't re-pay for the same queries.

---

## Why Claude and not DeepL?

DeepL's API Free requires a billing address from a supported country and a credit card issued there. Armenia, Russia, Belarus, and several others are not supported. Anthropic's API is available in most countries, cheaper at this scale once you factor in caching, and arguably better on short SEO text because it understands intent (shopping, comparison, informational) rather than treating queries as sentences.

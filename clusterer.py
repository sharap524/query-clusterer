#!/usr/bin/env python3
"""
Professional SEO Query Clusterer — v2

Pipeline (deterministic, no agent needed):
  1. load      — read queries from TXT/CSV into SQLite
  2. translate — batch-translate via Claude (also detects source language)
  3. cluster   — embed ENGLISH translations, agglomerative cosine clustering
  4. export    — dump JSON/CSV

Why translate before clustering?
  The old version embedded raw multilingual queries and relied on the
  multilingual model to align them. That works but is noisy on short
  SEO strings. Translating to English first gives (a) much cleaner
  cluster centroids and (b) human-readable cluster names for free.

Changes vs v1:
  - Real translation via Claude Haiku 4.5 (replaces 30-word dictionary).
  - Language detection via Claude (replaces langid, which was unreliable
    on 2-3 word SEO queries with brand names in Latin script).
  - Cluster names built from English translations, not token salad.
  - Translation persisted in DB → clustering is cheap to re-run at
    different thresholds without re-calling the API.
"""

import json
import os
import re
import sqlite3
import warnings
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from translator import ClaudeTranslator


class SEOQueryClusterer:
    """Professional SEO Query Clusterer."""

    # Multilingual model still useful as a fallback if translation fails,
    # but on translated (English) text, an English model would also work.
    # We keep multilingual so a mixed state (some translated, some not) is safe.
    DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

    LANG_NAMES = {
        "en": "English", "ru": "Russian", "de": "German", "fr": "French",
        "es": "Spanish", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
        "uk": "Ukrainian", "nl": "Dutch", "tr": "Turkish", "ja": "Japanese",
        "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
        "id": "Indonesian", "cs": "Czech", "sv": "Swedish", "da": "Danish",
        "nb": "Norwegian", "fi": "Finnish", "el": "Greek", "ro": "Romanian",
        "hu": "Hungarian", "bg": "Bulgarian", "sk": "Slovak", "sl": "Slovenian",
        "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
        "und": "Unknown",
    }

    # Minimal stopword set for cluster-name extraction (English only now —
    # we name clusters from translated text).
    EN_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "and", "but",
        "if", "or", "because", "until", "while", "what", "which", "who",
        "whom", "this", "that", "these", "those", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "how", "where", "when", "why",
    }

    def __init__(self, db_path: str = "seo_queries.db", model_name: str | None = None):
        self.db_path = db_path
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self._init_database()

    # ---------------------------------------------------------------- schema

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL UNIQUE,
                query_normalized TEXT,
                language TEXT,
                detected_by TEXT,          -- 'claude' | 'pending' | 'error'
                translation_en TEXT,
                translation_source TEXT,   -- 'claude' | 'cache' | 'passthrough' | 'error' | NULL
                cluster_id INTEGER,
                cluster_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY,
                name TEXT,
                name_slug TEXT,
                query_count INTEGER,
                languages TEXT,
                centroid_query TEXT,
                centroid_query_en TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cluster_id ON queries(cluster_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_language ON queries(language)")
        conn.commit()
        conn.close()

    # ---------------------------------------------------------------- load

    @staticmethod
    def normalize_query(query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r"\s+", " ", q)
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def add_queries(self, queries: list[str]) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        added = skipped = 0
        for query in queries:
            query = query.strip()
            if not query or len(query) < 2:
                continue
            normalized = self.normalize_query(query)
            cur.execute(
                "SELECT id FROM queries WHERE query = ? OR query_normalized = ?",
                (query, normalized),
            )
            if cur.fetchone():
                skipped += 1
                continue
            cur.execute(
                "INSERT INTO queries (query, query_normalized, language, detected_by) "
                "VALUES (?, ?, NULL, 'pending')",
                (query, normalized),
            )
            added += 1
        conn.commit()
        conn.close()
        print(f"✅ Added: {added} queries. Skipped duplicates: {skipped}")
        return added

    def load_from_file(self, filepath: str, column: str | None = None) -> int:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
            if column and column in df.columns:
                queries = df[column].dropna().astype(str).tolist()
            else:
                for col in ("query", "keyword", "keywords", "search_query",
                            "queries", "term", "terms"):
                    if col in df.columns:
                        queries = df[col].dropna().astype(str).tolist()
                        print(f"📄 Using column: {col}")
                        break
                else:
                    queries = df.iloc[:, 0].dropna().astype(str).tolist()
                    print(f"📄 Using first column: {df.columns[0]}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]

        return self.add_queries(queries)

    # ---------------------------------------------------------------- translate

    def translate_pending(
        self,
        api_key: str | None = None,
        cache_db: str = "translations.db",
        force: bool = False,
    ) -> dict:
        """
        Translate all queries in DB that don't yet have a translation
        (or all of them, if force=True).
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        if force:
            cur.execute("SELECT id, query FROM queries")
        else:
            cur.execute(
                "SELECT id, query FROM queries "
                "WHERE translation_en IS NULL OR translation_source IN ('error', 'passthrough')"
            )
        rows = cur.fetchall()

        if not rows:
            conn.close()
            print("✅ Nothing to translate — all queries already have translations.")
            return {"translated": 0, "cached": 0, "errors": 0}

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        translator = ClaudeTranslator(api_key=api_key, cache_db=cache_db)
        print(f"🔄 Translating {len(texts)} queries via Claude...")
        results = translator.translate_batch(texts, use_cache=True)

        # Persist back to main DB
        for qid, src, result in zip(ids, texts, results):
            cur.execute(
                "UPDATE queries SET "
                "language = ?, detected_by = ?, "
                "translation_en = ?, translation_source = ? "
                "WHERE id = ?",
                (
                    result.detected_lang,
                    "claude" if result.source in ("claude", "cache") else "error",
                    result.text,
                    result.source,
                    qid,
                ),
            )
        conn.commit()
        conn.close()

        stats = Counter(r.source for r in results)
        print(f"✅ Translated: {stats['claude']}, cached: {stats['cache']}, "
              f"errors: {stats['error'] + stats['passthrough']}")
        return dict(stats)

    # ---------------------------------------------------------------- cluster

    def _load_model(self):
        if self.model is None:
            print(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Model loaded")
        return self.model

    def _get_queries_for_clustering(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, query, language, translation_en, translation_source
            FROM queries
        """)
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "query": r[1],
                "language": r[2] or "und",
                "translation_en": r[3] or r[1],   # fall back to original
                "translation_source": r[4],
            }
            for r in rows
        ]

    def cluster(
        self,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 1,
        max_clusters: int | None = None,
        use_translations: bool = True,
    ) -> dict:
        """
        Hierarchical cosine clustering.

        use_translations: if True (default) and translations exist, embed
        the English translation instead of the raw query. This gives
        sharper clusters since all vectors live in the same linguistic
        space. If a query has no translation, its original text is used.
        """
        queries_data = self._get_queries_for_clustering()
        if len(queries_data) < 2:
            raise ValueError("Need at least 2 queries for clustering")

        if use_translations:
            texts_to_embed = [q["translation_en"] for q in queries_data]
            untranslated = sum(
                1 for q in queries_data
                if not q["translation_source"] or q["translation_source"] in ("error", "passthrough")
            )
            if untranslated:
                print(f"⚠️  {untranslated} queries have no Claude translation — "
                      f"using original text for those. Run `translate` first for best results.")
        else:
            texts_to_embed = [q["query"] for q in queries_data]

        model = self._load_model()
        print(f"🔄 Generating embeddings for {len(texts_to_embed)} queries...")
        embeddings = model.encode(
            texts_to_embed, show_progress_bar=True, normalize_embeddings=True
        )

        print("🔄 Building similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        # Clamp tiny negatives from floating-point drift; AgglomerativeClustering
        # with metric='precomputed' is fussy about this.
        np.clip(distance_matrix, 0, 2, out=distance_matrix)

        distance_threshold = 1 - similarity_threshold
        print(f"🔄 Clustering with similarity threshold {similarity_threshold}...")

        clustering = AgglomerativeClustering(
            n_clusters=max_clusters,
            distance_threshold=distance_threshold if max_clusters is None else None,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Group
        clusters_raw = defaultdict(list)
        for i, label in enumerate(labels):
            clusters_raw[label].append({
                "id": queries_data[i]["id"],
                "query": queries_data[i]["query"],
                "translation_en": queries_data[i]["translation_en"],
                "language": queries_data[i]["language"],
                "embedding_idx": i,
            })

        # Optional: absorb small clusters into nearest big one
        clusters = {}
        noise = []
        for label, items in clusters_raw.items():
            if len(items) >= min_cluster_size:
                clusters[label] = items
            else:
                noise.extend(items)

        if noise and clusters:
            centroids = {
                label: embeddings[[it["embedding_idx"] for it in items]].mean(axis=0)
                for label, items in clusters.items()
            }
            for n in noise:
                best, best_sim = None, 0.0
                emb = embeddings[n["embedding_idx"]]
                for label, c in centroids.items():
                    sim = float(np.dot(emb, c))
                    if sim > best_sim and sim >= similarity_threshold - 0.1:
                        best, best_sim = label, sim
                if best is not None:
                    clusters[best].append(n)
                else:
                    new_label = (max(clusters.keys()) + 1) if clusters else 0
                    clusters[new_label] = [n]

        print("🔄 Generating cluster metadata...")
        results = self._build_cluster_metadata(clusters, embeddings)
        self._save_results(results)

        silhouette = 0.0
        if len(set(labels)) > 1:
            silhouette = float(silhouette_score(distance_matrix, labels, metric="precomputed"))

        print(f"✅ Done: {len(results)} clusters")
        return {
            "total_queries": len(queries_data),
            "num_clusters": len(results),
            "similarity_threshold": similarity_threshold,
            "silhouette_score": round(silhouette, 3),
            "clusters": results,
        }

    def _build_cluster_metadata(self, clusters: dict, embeddings: np.ndarray) -> dict:
        results = {}
        for label, items in clusters.items():
            idxs = [it["embedding_idx"] for it in items]
            cluster_embs = embeddings[idxs]
            centroid = cluster_embs.mean(axis=0)
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            central_idx = int(distances.argmin())
            centroid_item = items[central_idx]

            lang_counts = Counter(it["language"] for it in items)
            primary_lang = lang_counts.most_common(1)[0][0]

            cluster_name = self._build_cluster_name(items, centroid_item)
            slug = re.sub(r"[^a-z0-9]+", "_", cluster_name.lower()).strip("_") or f"cluster_{label}"

            query_list = [
                {
                    "query": it["query"],
                    "language": it["language"],
                    "language_name": self.LANG_NAMES.get(it["language"], it["language"]),
                    "translation_en": it["translation_en"],
                }
                for it in items
            ]

            # Avoid name collisions between clusters
            final_name = cluster_name
            suffix = 2
            while final_name in results:
                final_name = f"{cluster_name} ({suffix})"
                suffix += 1

            results[final_name] = {
                "id": int(label),
                "slug": slug,
                "query_count": len(items),
                "languages": dict(lang_counts),
                "primary_language": primary_lang,
                "centroid_query": centroid_item["query"],
                "centroid_query_en": centroid_item["translation_en"],
                "queries": query_list,
            }

        return dict(sorted(results.items(), key=lambda x: -x[1]["query_count"]))

    def _build_cluster_name(self, items: list, centroid_item: dict) -> str:
        """
        Build a human-readable English cluster name from translated queries.
        Strategy: take top non-stopword tokens across all English translations,
        preserving order of appearance in the centroid query. This usually
        produces something like "Iphone 15 Pro Price" rather than a bag-of-words.
        """
        # Collect token frequencies across translations
        freq = Counter()
        for it in items:
            tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b", it["translation_en"].lower())
            for tok in tokens:
                if tok not in self.EN_STOPWORDS:
                    freq[tok] += 1

        if not freq:
            return (centroid_item["translation_en"] or centroid_item["query"])[:50].title()

        # Keep top 4 tokens, but ordered as they appear in centroid translation
        top_tokens = {tok for tok, _ in freq.most_common(4)}
        centroid_tokens = re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b", centroid_item["translation_en"].lower()
        )
        ordered = []
        seen = set()
        for tok in centroid_tokens:
            if tok in top_tokens and tok not in seen:
                ordered.append(tok)
                seen.add(tok)
        # If centroid translation didn't cover the top tokens, append missing ones by freq
        for tok, _ in freq.most_common(4):
            if tok not in seen:
                ordered.append(tok)
                seen.add(tok)

        name = " ".join(ordered[:4]).title()
        return name[:50] if len(name) > 50 else name

    # ---------------------------------------------------------------- persistence

    def _save_results(self, results: dict):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM clusters")
        cur.execute("UPDATE queries SET cluster_id = NULL, cluster_name = NULL")

        for cluster_name, data in results.items():
            cur.execute(
                "INSERT INTO clusters "
                "(id, name, name_slug, query_count, languages, centroid_query, centroid_query_en) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    data["id"], cluster_name, data["slug"], data["query_count"],
                    json.dumps(data["languages"]), data["centroid_query"],
                    data["centroid_query_en"],
                ),
            )
            for q in data["queries"]:
                cur.execute(
                    "UPDATE queries SET cluster_id = ?, cluster_name = ? WHERE query = ?",
                    (data["id"], cluster_name, q["query"]),
                )
        conn.commit()
        conn.close()

    # ---------------------------------------------------------------- export

    def export_json(self, output_path: str = "clustering_results.json") -> str:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, name_slug, query_count, languages, centroid_query, centroid_query_en "
            "FROM clusters ORDER BY query_count DESC"
        )
        clusters_raw = cur.fetchall()

        out = {
            "generated_at": datetime.now().isoformat(),
            "total_queries": 0,
            "num_clusters": len(clusters_raw),
            "clusters": {},
        }

        for cid, name, slug, count, langs_json, centroid, centroid_en in clusters_raw:
            cur.execute(
                "SELECT query, language, translation_en FROM queries "
                "WHERE cluster_id = ? ORDER BY query",
                (cid,),
            )
            queries = cur.fetchall()
            out["clusters"][name] = {
                "id": cid,
                "slug": slug,
                "query_count": count,
                "languages": json.loads(langs_json) if langs_json else {},
                "centroid_query": centroid,
                "centroid_query_en": centroid_en,
                "queries": [
                    {
                        "query": q[0],
                        "language": q[1],
                        "language_name": self.LANG_NAMES.get(q[1], q[1]),
                        "translation_en": q[2] or q[0],
                    }
                    for q in queries
                ],
            }
            out["total_queries"] += count

        conn.close()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"✅ Exported to {output_path}")
        return output_path

    def export_csv(self, output_path: str = "clustered_queries.csv") -> str:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT
                cluster_name  AS 'Cluster',
                cluster_id    AS 'Cluster ID',
                query         AS 'Query',
                language      AS 'Language',
                translation_en AS 'Translation (EN)',
                translation_source AS 'Translation Source'
            FROM queries
            WHERE cluster_id IS NOT NULL
            ORDER BY cluster_id, query
        """, conn)
        conn.close()

        df["Language Name"] = df["Language"].map(lambda x: self.LANG_NAMES.get(x, x))
        df = df[[
            "Cluster", "Cluster ID", "Query", "Language", "Language Name",
            "Translation (EN)", "Translation Source",
        ]]
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ Exported to {output_path}")
        return output_path

    # ---------------------------------------------------------------- stats

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM queries")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM queries WHERE cluster_id IS NOT NULL")
        clustered = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM queries WHERE translation_source = 'claude' OR translation_source = 'cache'")
        translated = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM clusters")
        num_clusters = cur.fetchone()[0]

        cur.execute(
            "SELECT language, COUNT(*) FROM queries "
            "GROUP BY language ORDER BY COUNT(*) DESC"
        )
        by_language = dict(cur.fetchall())

        cur.execute(
            "SELECT name, query_count FROM clusters "
            "ORDER BY query_count DESC LIMIT 10"
        )
        top_clusters = dict(cur.fetchall())

        conn.close()
        return {
            "total_queries": total,
            "translated_queries": translated,
            "clustered_queries": clustered,
            "num_clusters": num_clusters,
            "languages": by_language,
            "top_clusters": top_clusters,
        }


# ---------------------------------------------------------------- CLI

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Professional SEO Query Clusterer (Claude-powered)")
    parser.add_argument(
        "command",
        choices=["load", "translate", "cluster", "export", "stats", "all"],
        help="load → translate → cluster → export (or `all` to run the whole pipeline)",
    )
    parser.add_argument("--input", "-i", help="Input file (TXT or CSV)")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--db", default="seo_queries.db", help="Main DB path")
    parser.add_argument("--translation-cache", default="translations.db",
                        help="Claude translation cache DB")
    parser.add_argument("--threshold", "-t", type=float, default=0.75,
                        help="Similarity threshold 0.6–0.9 (higher = stricter)")
    parser.add_argument("--column", "-c", help="CSV column name")
    parser.add_argument("--anthropic-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env)")
    parser.add_argument("--force-retranslate", action="store_true",
                        help="Re-translate even queries that already have a translation")
    parser.add_argument("--no-translations", action="store_true",
                        help="Cluster on raw queries instead of English translations")

    args = parser.parse_args()
    clusterer = SEOQueryClusterer(db_path=args.db)

    if args.command == "load":
        if not args.input:
            print("❌ Specify input file with --input"); return
        clusterer.load_from_file(args.input, column=args.column)

    elif args.command == "translate":
        clusterer.translate_pending(
            api_key=args.anthropic_key,
            cache_db=args.translation_cache,
            force=args.force_retranslate,
        )

    elif args.command == "cluster":
        result = clusterer.cluster(
            similarity_threshold=args.threshold,
            use_translations=not args.no_translations,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "clusters"},
                         ensure_ascii=False, indent=2))

    elif args.command == "export":
        output = args.output or "clustering_results.json"
        if output.endswith(".csv"):
            clusterer.export_csv(output)
        else:
            clusterer.export_json(output)

    elif args.command == "stats":
        print(json.dumps(clusterer.get_stats(), ensure_ascii=False, indent=2))

    elif args.command == "all":
        if not args.input:
            print("❌ Specify input file with --input"); return
        clusterer.load_from_file(args.input, column=args.column)
        clusterer.translate_pending(
            api_key=args.anthropic_key,
            cache_db=args.translation_cache,
            force=args.force_retranslate,
        )
        clusterer.cluster(
            similarity_threshold=args.threshold,
            use_translations=not args.no_translations,
        )
        clusterer.export_json(args.output or "clustering_results.json")
        clusterer.export_csv((args.output or "clustering_results").replace(".json", "") + ".csv")
        print(json.dumps(clusterer.get_stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

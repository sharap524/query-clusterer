#!/usr/bin/env python3
"""
Query Clusterer - Кластеризация поисковых запросов по семантике и языкам
Использует бесплатные модели и базы данных для работы в GitHub Actions
"""

import json
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import langdetect
from langdetect import detect, detect_langs
import pandas as pd


class QueryClusterer:
    """Кластеризатор запросов с определением языка и семантической группировкой"""
    
    def __init__(self, db_path: str = "queries.db", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализация кластеризатора
        
        Args:
            db_path: Путь к SQLite базе данных
            model_name: Название модели для эмбеддингов (бесплатная multilingual модель)
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self._init_database()
    
    def _init_database(self):
        """Инициализация SQLite базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица запросов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                language TEXT,
                cluster_id INTEGER,
                cluster_name TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица кластеров
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                language TEXT,
                query_count INTEGER DEFAULT 0,
                representative_query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица результатов кластеризации
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clustering_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_queries INTEGER,
                num_clusters INTEGER,
                silhouette_score REAL,
                params TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ База данных инициализирована: {self.db_path}")
    
    def _load_model(self):
        """Ленивая загрузка модели"""
        if self.model is None:
            print(f"📥 Загрузка модели {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Модель загружена")
        return self.model
    
    def detect_language(self, text: str) -> dict:
        """
        Определение языка текста
        
        Returns:
            dict с языком и уверенностью
        """
        try:
            langs = detect_langs(text)
            primary = langs[0]
            return {
                "language": primary.lang,
                "confidence": primary.prob,
                "all_detected": [(l.lang, round(l.prob, 3)) for l in langs[:3]]
            }
        except Exception:
            return {"language": "unknown", "confidence": 0.0, "all_detected": []}
    
    def add_queries(self, queries: list[str]) -> int:
        """
        Добавление запросов в базу данных
        
        Args:
            queries: Список текстовых запросов
            
        Returns:
            Количество добавленных запросов
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        added = 0
        for query in queries:
            query = query.strip()
            if not query:
                continue
            
            # Определяем язык
            lang_info = self.detect_language(query)
            
            cursor.execute(
                "INSERT INTO queries (query, language) VALUES (?, ?)",
                (query, lang_info["language"])
            )
            added += 1
        
        conn.commit()
        conn.close()
        print(f"✅ Добавлено {added} запросов")
        return added
    
    def load_queries_from_file(self, filepath: str) -> int:
        """Загрузка запросов из файла (по одному на строку)"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        
        return self.add_queries(queries)
    
    def load_queries_from_csv(self, filepath: str, column: str = "query") -> int:
        """Загрузка запросов из CSV файла"""
        df = pd.read_csv(filepath)
        if column not in df.columns:
            raise ValueError(f"Колонка '{column}' не найдена. Доступные: {list(df.columns)}")
        
        queries = df[column].dropna().astype(str).tolist()
        return self.add_queries(queries)
    
    def get_all_queries(self) -> list[dict]:
        """Получение всех запросов из базы"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, query, language, cluster_id, cluster_name FROM queries")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"id": r[0], "query": r[1], "language": r[2], "cluster_id": r[3], "cluster_name": r[4]}
            for r in rows
        ]
    
    def cluster_queries(
        self,
        n_clusters: int = None,
        min_cluster_size: int = 3,
        algorithm: str = "kmeans"
    ) -> dict:
        """
        Кластеризация запросов
        
        Args:
            n_clusters: Количество кластеров (авто если None)
            min_cluster_size: Минимальный размер кластера для DBSCAN
            algorithm: 'kmeans' или 'dbscan'
            
        Returns:
            Результаты кластеризации
        """
        # Получаем запросы
        queries_data = self.get_all_queries()
        if len(queries_data) < 2:
            raise ValueError("Нужно минимум 2 запроса для кластеризации")
        
        queries = [q["query"] for q in queries_data]
        query_ids = [q["id"] for q in queries_data]
        
        # Создаём эмбеддинги
        model = self._load_model()
        print(f"🔄 Создание эмбеддингов для {len(queries)} запросов...")
        embeddings = model.encode(queries, show_progress_bar=True)
        
        # Автоматический выбор количества кластеров
        if n_clusters is None and algorithm == "kmeans":
            n_clusters = self._find_optimal_clusters(embeddings, max_k=min(20, len(queries) // 2))
        
        # Кластеризация
        print(f"🔄 Кластеризация методом {algorithm}...")
        if algorithm == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
        elif algorithm == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size, metric="cosine")
            labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Неизвестный алгоритм: {algorithm}")
        
        # Вычисляем метрики
        unique_labels = set(labels) - {-1}  # -1 это шум в DBSCAN
        silhouette = silhouette_score(embeddings, labels) if len(unique_labels) > 1 else 0
        
        # Группируем по кластерам
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append({
                "id": query_ids[i],
                "query": queries[i],
                "language": queries_data[i]["language"]
            })
        
        # Генерируем имена кластеров
        cluster_names = self._generate_cluster_names(clusters, embeddings, labels, queries)
        
        # Сохраняем результаты в базу
        self._save_clustering_results(query_ids, labels, cluster_names)
        
        # Сохраняем метаданные прогона
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO clustering_runs 
               (total_queries, num_clusters, silhouette_score, params) 
               VALUES (?, ?, ?, ?)""",
            (len(queries), len(unique_labels), silhouette, json.dumps({
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "min_cluster_size": min_cluster_size
            }))
        )
        conn.commit()
        conn.close()
        
        # Формируем результат
        result = {
            "total_queries": len(queries),
            "num_clusters": len(unique_labels),
            "silhouette_score": round(silhouette, 3),
            "algorithm": algorithm,
            "clusters": {}
        }
        
        for label, items in sorted(clusters.items()):
            if label == -1:
                name = "🔇 Шум (не кластеризовано)"
            else:
                name = cluster_names.get(label, f"Кластер {label}")
            
            # Группируем по языкам внутри кластера
            by_language = defaultdict(list)
            for item in items:
                by_language[item["language"]].append(item["query"])
            
            result["clusters"][name] = {
                "count": len(items),
                "languages": dict(by_language),
                "queries": [item["query"] for item in items[:10]]  # Первые 10 для превью
            }
        
        print(f"✅ Кластеризация завершена: {len(unique_labels)} кластеров, silhouette={silhouette:.3f}")
        return result
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """Поиск оптимального количества кластеров методом silhouette"""
        best_k = 2
        best_score = -1
        
        for k in range(2, min(max_k, len(embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"📊 Оптимальное количество кластеров: {best_k} (silhouette={best_score:.3f})")
        return best_k
    
    def _generate_cluster_names(
        self,
        clusters: dict,
        embeddings: np.ndarray,
        labels: np.ndarray,
        queries: list[str]
    ) -> dict:
        """Генерация понятных имён для кластеров"""
        names = {}
        
        for label, items in clusters.items():
            if label == -1:
                continue
            
            # Находим самый центральный запрос в кластере
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = cluster_embeddings.mean(axis=0)
            
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            central_idx = cluster_indices[distances.argmin()]
            
            # Берём ключевые слова из центрального и частых запросов
            central_query = queries[central_idx]
            
            # Определяем основной язык кластера
            langs = [item["language"] for item in items]
            main_lang = max(set(langs), key=langs.count)
            
            # Формируем имя
            lang_emoji = {
                "ru": "🇷🇺", "en": "🇬🇧", "de": "🇩🇪", "fr": "🇫🇷",
                "es": "🇪🇸", "it": "🇮🇹", "pt": "🇵🇹", "zh": "🇨🇳",
                "ja": "🇯🇵", "ko": "🇰🇷", "ar": "🇸🇦", "hi": "🇮🇳"
            }.get(main_lang, "🌐")
            
            # Берём первые слова центрального запроса как имя
            short_name = " ".join(central_query.split()[:4])
            if len(short_name) > 40:
                short_name = short_name[:37] + "..."
            
            names[label] = f"{lang_emoji} {short_name}"
        
        return names
    
    def _save_clustering_results(self, query_ids: list, labels: list, cluster_names: dict):
        """Сохранение результатов кластеризации в базу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for qid, label in zip(query_ids, labels):
            name = cluster_names.get(label, f"Кластер {label}" if label != -1 else "Шум")
            cursor.execute(
                "UPDATE queries SET cluster_id = ?, cluster_name = ? WHERE id = ?",
                (int(label), name, qid)
            )
        
        conn.commit()
        conn.close()
    
    def export_results(self, output_path: str = "clustering_results.json"):
        """Экспорт результатов в JSON"""
        queries = self.get_all_queries()
        
        # Группируем по кластерам
        by_cluster = defaultdict(lambda: {"queries": [], "languages": defaultdict(int)})
        
        for q in queries:
            cluster = q["cluster_name"] or "Не кластеризовано"
            by_cluster[cluster]["queries"].append(q["query"])
            by_cluster[cluster]["languages"][q["language"]] += 1
        
        result = {
            "generated_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "clusters": {
                name: {
                    "count": len(data["queries"]),
                    "languages": dict(data["languages"]),
                    "sample_queries": data["queries"][:20]
                }
                for name, data in sorted(by_cluster.items(), key=lambda x: -len(x[1]["queries"]))
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Результаты экспортированы: {output_path}")
        return output_path
    
    def export_csv(self, output_path: str = "clustered_queries.csv"):
        """Экспорт в CSV"""
        queries = self.get_all_queries()
        df = pd.DataFrame(queries)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ CSV экспортирован: {output_path}")
        return output_path
    
    def get_statistics(self) -> dict:
        """Получение статистики по базе"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Общее количество
        cursor.execute("SELECT COUNT(*) FROM queries")
        total = cursor.fetchone()[0]
        
        # По языкам
        cursor.execute("SELECT language, COUNT(*) FROM queries GROUP BY language ORDER BY COUNT(*) DESC")
        by_language = dict(cursor.fetchall())
        
        # По кластерам
        cursor.execute("SELECT cluster_name, COUNT(*) FROM queries WHERE cluster_name IS NOT NULL GROUP BY cluster_name ORDER BY COUNT(*) DESC")
        by_cluster = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_queries": total,
            "by_language": by_language,
            "by_cluster": by_cluster,
            "num_languages": len(by_language),
            "num_clusters": len(by_cluster)
        }


def main():
    """CLI интерфейс"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Кластеризатор запросов")
    parser.add_argument("command", choices=["add", "cluster", "export", "stats", "load-csv", "load-txt"])
    parser.add_argument("--input", "-i", help="Входной файл")
    parser.add_argument("--output", "-o", help="Выходной файл")
    parser.add_argument("--db", default="queries.db", help="Путь к базе данных")
    parser.add_argument("--n-clusters", "-n", type=int, help="Количество кластеров")
    parser.add_argument("--algorithm", "-a", default="kmeans", choices=["kmeans", "dbscan"])
    parser.add_argument("--column", "-c", default="query", help="Колонка с запросами в CSV")
    parser.add_argument("--queries", nargs="+", help="Запросы для добавления")
    
    args = parser.parse_args()
    
    clusterer = QueryClusterer(db_path=args.db)
    
    if args.command == "add":
        if args.queries:
            clusterer.add_queries(args.queries)
        else:
            print("Укажите запросы через --queries")
    
    elif args.command == "load-txt":
        if not args.input:
            print("Укажите входной файл через --input")
            return
        clusterer.load_queries_from_file(args.input)
    
    elif args.command == "load-csv":
        if not args.input:
            print("Укажите входной файл через --input")
            return
        clusterer.load_queries_from_csv(args.input, column=args.column)
    
    elif args.command == "cluster":
        result = clusterer.cluster_queries(
            n_clusters=args.n_clusters,
            algorithm=args.algorithm
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.command == "export":
        output = args.output or "clustering_results.json"
        if output.endswith(".csv"):
            clusterer.export_csv(output)
        else:
            clusterer.export_results(output)
    
    elif args.command == "stats":
        stats = clusterer.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

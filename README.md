# 🔍 Кластеризатор запросов

Автоматически группирует запросы по смыслу и языкам.

---

## 🚀 Как запустить (5 шагов)

### 1. Создай репозиторий

Открой https://github.com/new и создай новый репозиторий.

### 2. Загрузи файлы

На странице репозитория нажми **"uploading an existing file"** и загрузи эти файлы:

```
clusterer.py
requirements.txt
queries.txt
.github/workflows/cluster.yml
```

**Важно:** папка `.github` с файлом внутри — это один файл при загрузке. GitHub сам создаст структуру папок.

### 3. Замени запросы на свои

1. Открой файл `queries.txt`
2. Нажми карандаш ✏️ (Edit)
3. Удали примеры, вставь свои запросы (каждый на новой строке)
4. Нажми **Commit changes**

### 4. Запусти кластеризацию

1. Вкладка **Actions**
2. Слева выбери **Query Clustering**
3. Нажми **Run workflow** → **Run workflow**
4. Жди 2-3 минуты (появится ✅)

### 5. Скачай результаты

1. Кликни на завершённый запуск
2. Внизу в **Artifacts** скачай архив
3. Внутри:
   - `clustering_results.json` — все кластеры
   - `clustered_queries.csv` — таблица для Excel
   - `REPORT.md` — отчёт текстом

---

## ❓ Проблемы

**Actions не работает**
→ Settings → Actions → General → выбери "Allow all actions"

**Ошибка "file not found"**
→ Проверь что `queries.txt` загружен

**Хочу 5 кластеров**
→ При запуске в поле `n_clusters` напиши `5`

---

## 📝 Формат файла запросов

**queries.txt** — по одному запросу на строку:
```
купить кроссовки
buy sneakers online
где купить найки
nike shoes price
```

**queries.csv** — с заголовком `query`:
```
query
купить кроссовки
buy sneakers online
```

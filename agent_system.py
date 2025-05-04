from typing import List, Dict, Any, Optional, Type
import os
import json
from datetime import datetime, timedelta
from langchain.tools import Tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from pydantic import Field
from dotenv import load_dotenv
import feedparser
import requests
import sqlite3

# Імпортуємо джерела новин
from news_sources import (
    ALL_SCRAPING_URLS, 
    UKRAINIAN_SCRAPING_URLS, 
    INTERNATIONAL_SCRAPING_URLS
)

# Завантаження змінних середовища
load_dotenv()

# Конфігурація системи
class NewsConfig:
    def __init__(self, rss_feeds=None, categories=None, ai_queries=None, db_path=None, output_file=None):
        self.rss_feeds = rss_feeds or []
        self.categories = categories or {}
        self.ai_queries = ai_queries or []
        self.db_path = db_path or os.getenv("DB_PATH", "articles.db")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.output_file = output_file or os.getenv("OUTPUT_FILE", "news_report.html")
        
        # Перевірка наявності ключа API
        if not self.openai_api_key:
            print("УВАГА: API ключ OpenAI не знайдено. Деякі функції будуть недоступні.")

# --- Інструменти ---
class RSSFeedTool(BaseTool):
    name: str = "fetch_rss"
    description: str = "Отримує новини з RSS-стрічок за заданим списком URL. Корисно для отримання найсвіжіших новин з технологічних сайтів."
    rss_feeds: List[str] = Field(default_factory=list)
    
    def __init__(self, rss_feeds: List[str]):
        super().__init__()
        self.rss_feeds = rss_feeds
    
    def _run(self, urls: str = None) -> str:
        """Виконання інструменту"""
        feed_urls = self.rss_feeds
        if urls:
            try:
                additional_urls = urls.split(',')
                feed_urls.extend([url.strip() for url in additional_urls])
            except:
                pass
        
        # Визначаємо дату тижневої давності
        week_ago = datetime.now() - timedelta(days=7)
        
        results = []
        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:  # Обмежуємо до 5 записів на канал
                    # Парсимо дату публікації
                    pub_date = None
                    if 'published_parsed' in entry:
                        pub_struct = entry.published_parsed
                        pub_date = datetime(*pub_struct[:6])
                    elif 'updated_parsed' in entry:
                        pub_struct = entry.updated_parsed
                        pub_date = datetime(*pub_struct[:6])
                    
                    # Пропускаємо статті старші за тиждень
                    if pub_date and pub_date < week_ago:
                        continue
                    
                    results.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', ''),
                        'source': f"RSS: {url}",
                        'pub_date': pub_date.isoformat() if pub_date else datetime.now().isoformat()  # Зберігаємо дату в ISO форматі
                    })
            except Exception as e:
                results.append({"error": f"Помилка при читанні {url}: {str(e)}"})
        
        print(f"Знайдено {len(results)} актуальних статей (за останній тиждень)")
        return json.dumps(results, ensure_ascii=False, indent=2)

class NewsLLMTool(BaseTool):
    name: str = "search_ai_news"
    description: str = "Отримує останні новини про штучний інтелект за допомогою LLM. Використовуй, коли потрібно знайти останні тренди або події у сфері AI."
    openai_api_key: Optional[str] = None
    llm: Optional[Any] = None
    
    def __init__(self, openai_api_key: str = None):
        super().__init__()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-4o",
                temperature=0.2
            )
    
    def _run(self, query: str = "Останні новини про штучний інтелект") -> str:
        """Виконання інструменту"""
        if not self.openai_api_key:
            print("ПОМИЛКА: API ключ OpenAI не налаштовано. Перевірте наявність OPENAI_API_KEY в середовищі.")
            return json.dumps([])
        
        prompt = f"""Знайди останні важливі новини на тему: "{query}".
        Для кожної новини наведи:
        1. Заголовок
        2. Короткий опис (3-4 речення)
        3. Джерело і дату (якщо відомо)
        
        Формат відповіді: JSON масив, де кожна новина має поля title, summary, source, date.
        Відповідай строго у форматі JSON і нічого більше.
        Приклад:
        [
            {{
                "title": "Заголовок новини 1",
                "summary": "Короткий опис новини 1",
                "source": "Джерело 1",
                "date": "01.05.2023"
            }},
            {{
                "title": "Заголовок новини 2",
                "summary": "Короткий опис новини 2",
                "source": "Джерело 2",
                "date": "02.05.2023"
            }}
        ]
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Обробка відповіді для вилучення лише JSON
            import re
            # Різні шаблони для пошуку JSON у відповіді
            json_patterns = [
                r'\[\s*\{.*\}\s*\]',  # Простий масив об'єктів
                r'```json\s*([\s\S]*?)```',  # JSON у блоці коду
                r'```\s*([\s\S]*?)```',  # Будь-який блок коду
            ]
            
            json_content = None
            # Спробуємо різні шаблони для пошуку JSON
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0).strip()
                    # Якщо знайшли у блоці коду, видаляємо маркери блоку
                    if pattern.startswith('```'):
                        json_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_content, flags=re.DOTALL)
                    break
            
            # Якщо не знайшли JSON за шаблонами, використовуємо всю відповідь
            if not json_content:
                json_content = content
            
            # Перевіряємо, чи можемо розпарсити JSON
            try:
                json.loads(json_content)
                return json_content
            except json.JSONDecodeError as e:
                # Спроба виправити найпоширеніші проблеми
                # 1. Замінюємо одинарні лапки на подвійні
                fixed_content = json_content.replace("'", "\"")
                # 2. Виправляємо проблеми із зайвими комами в кінці масивів/об'єктів
                fixed_content = re.sub(r',\s*}', '}', fixed_content)
                fixed_content = re.sub(r',\s*\]', ']', fixed_content)
                
                try:
                    json.loads(fixed_content)
                    return fixed_content
                except json.JSONDecodeError:
                    # Якщо не змогли розпарсити JSON, повертаємо оригінальну відповідь
                    return f"Помилка: неможливо отримати валідний JSON. Оригінальна відповідь: {content[:100]}..."
        except Exception as e:
            return f"Помилка отримання новин через LLM: {str(e)}"

class TranslationTool(BaseTool):
    name: str = "translate_text"
    description: str = "Перекладає текст українською мовою. Використовуй для перекладу заголовків і описів статей."
    openai_api_key: Optional[str] = None
    llm: Optional[Any] = None
    
    def __init__(self, openai_api_key: str = None):
        super().__init__()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
    
    def _run(self, text: str) -> str:
        """Виконання інструменту"""
        if not self.openai_api_key or not text or len(text.strip()) < 5:
            return text
        
        prompt = f"""Переклади наступний текст українською мовою, зберігаючи оригінальний формат і стиль:
        
        {text}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Помилка перекладу: {str(e)}"

class CategorizeTool(BaseTool):
    name: str = "categorize_article"
    description: str = "Категоризує статтю за її заголовком і змістом. Використовуй для визначення тематики новини."
    categories: Dict[str, List[str]] = Field(default_factory=dict)
    openai_api_key: Optional[str] = None
    llm: Optional[Any] = None
    
    def __init__(self, categories: Dict[str, List[str]], openai_api_key: str = None):
        super().__init__()
        self.categories = categories
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
    
    def _run(self, article_json: str) -> str:
        """Виконання інструменту"""
        try:
            article = json.loads(article_json)
        except:
            return "Помилка: неправильний формат JSON."
        
        if not self.openai_api_key:
            # Простий fallback на ключові слова
            for cat, keywords in self.categories.items():
                if any(kw.lower() in article.get('title', '').lower() for kw in keywords):
                    return cat
            return 'Інше'
        
        # Підготовка даних
        title = article.get('title', '')
        summary = article.get('summary', '')
                
        # Категорії у форматі для LLM
        categories_text = ", ".join(self.categories.keys())
        
        prompt = f"""Визнач категорію для наступної статті. Обери ЛИШЕ ОДНУ категорію із запропонованих: {categories_text}
        Якщо стаття не підходить для жодної з категорій, вкажи 'Інше'.
        
        Стаття:
        Заголовок: {title}
        Опис: {summary}
        
        Категорія (лише назва):
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            # Перевіряємо, чи відповідь є валідною категорією
            for category in self.categories.keys():
                if category.lower() in result.lower():
                    return category
            
            return 'Інше'
        except Exception as e:
            return f"Помилка категоризації: {str(e)}"

class SummarizeTool(BaseTool):
    name: str = "summarize_article"
    description: str = "Створює короткий дайджест статті. Використовуй для узагальнення довгих текстів."
    openai_api_key: Optional[str] = None
    llm: Optional[Any] = None
    
    def __init__(self, openai_api_key: str = None):
        super().__init__()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.3
            )
    
    def _run(self, text: str) -> str:
        """Виконання інструменту"""
        if not self.openai_api_key or not text or len(text.strip()) < 10:
            return "Недостатньо даних для узагальнення"
        
        prompt = f"""Узагальни наступний текст у 3-4 речення. Виділи найважливішу інформацію:
        
        {text}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Помилка узагальнення: {str(e)}"

class StorageTool(BaseTool):
    name: str = "store_articles"
    description: str = "Зберігає статті у базу даних SQLite. Використовуй для зберігання опрацьованих новин."
    db_path: str = "articles.db"
    
    def __init__(self, db_path: str = None):
        super().__init__()
        self.db_path = db_path or os.getenv("DB_PATH", "articles.db")
        self.create_table()
    
    def create_table(self):
        """Створює таблицю статей, якщо вона не існує"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            published TEXT,
            summary TEXT,
            ai_summary TEXT,
            category TEXT,
            source TEXT,
            ai_generated BOOLEAN DEFAULT 0,
            status TEXT
        )''')
        conn.commit()
        conn.close()
    
    def get_existing_links(self) -> List[str]:
        """Отримання списку всіх посилань, що вже є в базі даних"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT link FROM articles")
        links = [row[0] for row in cursor.fetchall()]
        conn.close()
        return links
    
    def _run(self, articles_json: str) -> str:
        """Виконання інструменту"""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                articles = json.loads(articles_json)
                
                # Перевірка на порожній масив
                if articles == []:
                    return "Немає статей для збереження."
                
            except json.JSONDecodeError as e:
                # Спроба розпізнати строку з повідомленням про помилку
                if "Помилка: неможливо отримати валідний JSON" in articles_json:
                    return f"Помилка при збереженні статей: неможливо розпарсити JSON"
                
                # Спробуємо знайти і виділити JSON, якщо він присутній в середині тексту
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', articles_json, re.DOTALL)
                if json_match:
                    try:
                        articles = json.loads(json_match.group(0))
                    except:
                        return f"Помилка при збереженні статей: некоректний формат даних"
                else:
                    # Перевірка на порожній масив в текстовому форматі
                    if articles_json.strip() == "[]":
                        return "Немає статей для збереження."
                        
                    return f"Помилка при збереженні статей: некоректний формат даних"
            
            if not isinstance(articles, list):
                articles = [articles]
            
            # Отримуємо список існуючих посилань для перевірки дублікатів
            existing_links = set(self.get_existing_links())
            
            new_count = 0
            skipped_count = 0
            
            for article in articles:
                try:
                    link = article.get('link', '')
                    
                    # Пропускаємо статті без посилань або які вже є в базі
                    if not link or link in existing_links:
                        skipped_count += 1
                        continue
                    
                    conn.execute('''INSERT OR IGNORE INTO articles 
                        (title, link, published, summary, ai_summary, category, source, ai_generated, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (article.get('title', ''), link, article.get('published', ''), 
                        article.get('summary', ''), article.get('ai_summary', ''), 
                        article.get('category', 'Інше'), article.get('source', 'невідомо'),
                        1 if article.get('ai_generated', False) else 0, 'new'))
                    new_count += 1
                    
                    # Додаємо посилання до списку існуючих для подальших перевірок
                    existing_links.add(link)
                    
                except Exception as e:
                    print(f"Помилка при збереженні статті: {str(e)}")
            
            conn.commit()
            conn.close()
            
            return f"Збережено {new_count} нових статей, пропущено {skipped_count} існуючих статей."
        except Exception as e:
            return f"Помилка при збереженні статей: {str(e)}"
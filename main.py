import feedparser
import requests
import sqlite3
import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from langchain.tools import Tool, BaseTool
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Dict, Any, Optional, Union, Set, Type
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import time
import uuid
import logging
import sqlite3
import requests
import traceback
import feedparser
import threading
import textwrap
import langchain
from tqdm import tqdm
import concurrent.futures
from langchain.agents import AgentExecutor, create_openai_tools_agent, initialize_agent, AgentType
from langchain.schema import SystemMessage
from pydantic import Field
from tavily import TavilyClient
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from logger_config import log_agent_action, get_logger
from pydantic import Field
from tavily import TavilyClient

# Імпортуємо логер
from logger_config import get_logger, log_agent_action, log_token_usage, print_token_usage_summary

# Імпортуємо джерела новин
from news_sources import (
    ALL_SCRAPING_URLS, 
    UKRAINIAN_SCRAPING_URLS, 
    INTERNATIONAL_SCRAPING_URLS,
    IT_TECH_SOURCES_UA, GENERAL_MEDIA_TECH_SECTIONS_UA,
    INTERNATIONAL_TECH_SOURCES, ALL_SOURCES
)

# Завантаження змінних середовища з .env
load_dotenv()

# Головний логер додатку
logger = get_logger()

# --- Agent Interfaces ---
class Agent:
    """Базовий клас для всіх агентів системи"""
    
    def run(self, *args, **kwargs):
        """Виконати агента синхронно"""
        raise NotImplementedError("Метод run() має бути реалізований у підкласі")
    
    async def arun(self, *args, **kwargs):
        """Виконати агента асинхронно"""
        raise NotImplementedError("Метод arun() має бути реалізований у підкласі")

# --- CrawlerAgent ---
class CrawlerAgent(Agent):
    def __init__(self, rss_feeds: List[str] = None):
        """Ініціалізація агента для збору новин з RSS-потоків"""
        self.logger = get_logger("crawler_agent")
        
        # Список RSS-потоків
        self.rss_feeds = rss_feeds or [
            'https://feeds.bbci.co.uk/news/technology/rss.xml',
            'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
            'https://techcrunch.com/feed/',
            'https://www.theverge.com/rss/index.xml',
            'https://ain.ua/feed/'
        ]
        
        self.logger.info(f"CrawlerAgent ініціалізовано з {len(self.rss_feeds)} RSS-потоками")
    
    def fetch_rss(self, url: str) -> List[Dict[str, Any]]:
        """Отримання статей з RSS-потоку"""
        log_agent_action("CrawlerAgent", "fetch_rss", f"Отримання RSS-потоку: {url}")
        try:
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:10]:  # Обмежуємо до 10 статей з кожного потоку
                article = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'content': entry.get('content', [{}])[0].get('value', '') if 'content' in entry else entry.get('summary', ''),
                    'source': f"RSS: {feed.feed.get('title', url)}"
                }
                
                # Додаємо поточну дату для сортування
                article['pub_date'] = datetime.now().isoformat()
                
                articles.append(article)
            
            self.logger.info(f"Отримано {len(articles)} статей з {url}")
            return articles
        except Exception as e:
            self.logger.error(f"Помилка при отриманні RSS-потоку {url}: {e}")
            return []
    
    async def fetch_rss_async(self, url: str) -> List[Dict[str, Any]]:
        """Асинхронне отримання статей з RSS-потоку"""
        log_agent_action("CrawlerAgent", "fetch_rss_async", f"Асинхронне отримання RSS-потоку: {url}")
        try:
            # Запускаємо синхронний метод в окремому потоці
            return await asyncio.to_thread(self.fetch_rss, url)
        except Exception as e:
            self.logger.error(f"Помилка при асинхронному отриманні RSS-потоку {url}: {e}")
            return []
    
    def run(self) -> List[Dict[str, Any]]:
        """Синхронне отримання статей з усіх RSS-потоків"""
        log_agent_action("CrawlerAgent", "run", f"Отримання новин з {len(self.rss_feeds)} RSS-потоків")
        start_time = time.time()
        
        all_articles = []
        for url in self.rss_feeds:
            articles = self.fetch_rss(url)
            all_articles.extend(articles)
        
        end_time = time.time()
        self.logger.info(f"Отримано всього {len(all_articles)} статей з {len(self.rss_feeds)} RSS-потоків за {end_time - start_time:.2f} секунд")
        
        return all_articles
    
    async def arun(self) -> List[Dict[str, Any]]:
        """Асинхронне отримання статей з усіх RSS-потоків"""
        log_agent_action("CrawlerAgent", "arun", f"Асинхронне отримання новин з {len(self.rss_feeds)} RSS-потоків")
        start_time = time.time()
        
        # Створюємо завдання для кожного RSS-потоку
        tasks = [self.fetch_rss_async(url) for url in self.rss_feeds]
        
        # Запускаємо всі завдання паралельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Об'єднуємо результати
        all_articles = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Помилка при отриманні RSS-потоку: {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
        
        end_time = time.time()
        self.logger.info(f"Асинхронно отримано всього {len(all_articles)} статей з {len(self.rss_feeds)} RSS-потоків за {end_time - start_time:.2f} секунд")
        
        return all_articles

# --- TranslatorAgent ---
class TranslatorAgent(Agent):
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or os.getenv("TRANSLATOR_API_URL")
        self.api_key = api_key or os.getenv("TRANSLATOR_API_KEY")
        self.logger = get_logger("translator_agent")
        
        # OpenAI API для перекладу
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
            # Додаємо змінну llm для сумісності зі старим кодом
            self.llm = self.chat_model
        
        self.logger.info("TranslatorAgent ініціалізовано")
        if self.api_url and self.api_key:
            self.logger.debug("Налаштовано зовнішнє API для перекладу")
        if self.openai_api_key:
            self.logger.debug("Налаштовано переклад через OpenAI")
            
    def detect_language(self, text: str) -> str:
        """Визначення мови тексту"""
        # Простий алгоритм для виявлення мови (українська або англійська)
        uk_chars = set('їієґ')
        
        # Конвертуємо текст у нижній регістр і видаляємо пробіли
        text_lower = text.lower()
        
        # Якщо є будь-який український символ, повертаємо 'uk'
        if any(char in uk_chars for char in text_lower):
            return 'uk'
            
        # Аналізуємо перші 100 символів для визначення мови
        text_sample = text_lower[:100]
        
        # Підрахунок українських та латинських символів
        uk_count = sum(1 for c in text_sample if 'а' <= c <= 'я' or c in uk_chars)
        en_count = sum(1 for c in text_sample if 'a' <= c <= 'z')
        
        # Якщо більше українських символів, повертаємо 'uk', інакше 'en'
        return 'uk' if uk_count > en_count else 'en'

    def translate_with_llm(self, text: str, target_lang: str = 'uk') -> str:
        """Переклад тексту з використанням мовної моделі"""
        if not self.openai_api_key or not self.llm:
            return text
            
        if not text or len(text.strip()) < 5:
            return text
        
        source_lang = self.detect_language(text)
        
        # Якщо текст вже українською, повертаємо як є
        if source_lang == 'uk' and target_lang == 'uk':
            return text
            
        # Промпт для перекладу англійською -> українською
        uk_prompt = f"""Переклади наступний текст українською мовою. 
        Переклад повинен бути природним, читабельним та адаптованим для українського читача.
        
        - Зберігай технічні терміни відповідно до загальноприйнятої термінології.
        - Для IT-термінів використовуй професійну термінологію (напр. "фреймворк", "бібліотека", "API").
        - Використовуй природні для української мови конструкції речень, уникай калькування з англійської.
        - Адаптуй сталі вирази та ідіоми, не перекладай їх дослівно.
        
        Текст для перекладу:
        {text}
        
        Переклад українською:
        """
        
        # Промпт для перекладу українською -> англійською
        en_prompt = f"""Translate the following text to English.
        The translation should sound natural and be adapted for English readers.
        
        Text to translate:
        {text}
        
        English translation:
        """
        
        try:
            if target_lang == 'uk':
                log_agent_action("TranslatorAgent", "translate_llm", f"Переклад тексту ({len(text)} символів) з {source_lang} на українську")
                prompt = uk_prompt
            else:
                log_agent_action("TranslatorAgent", "translate_llm", f"Переклад тексту ({len(text)} символів) з {source_lang} на англійську")
                prompt = en_prompt
                
            response = self.llm.invoke(prompt)
            
            # Логування використання токенів
            if hasattr(response, 'usage') and response.usage:
                log_token_usage({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'model': self.llm.model_name
                })
                
            return response.content
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    async def translate_with_llm_async(self, text: str, target_lang: str = 'uk') -> str:
        """Асинхронний переклад через OpenAI API"""
        log_agent_action("TranslatorAgent", "translate_with_llm_async", f"Асинхронний переклад тексту ({len(text)} символів) на {target_lang}")
        
        try:
            if not self.openai_api_key or not text or len(text.strip()) < 10:
                if not self.openai_api_key:
                    self.logger.warning("OpenAI API ключ не налаштовано")
                return text
                
            # Підготовка покращеного промпту для перекладу
            system_prompt = """Ти професійний перекладач. Твоє завдання - перекласти текст українською мовою, 
зберігаючи всі деталі, технічні терміни та стиль оригіналу. 
Переклад має бути природним і точним, з правильною українською лексикою, граматикою та пунктуацією.

Для технічних термінів використовуй усталені українські відповідники:
- artificial intelligence → штучний інтелект
- machine learning → машинне навчання
- deep learning → глибоке навчання
- neural network → нейронна мережа
- dataset → набір даних
- cloud computing → хмарні обчислення
- big data → великі дані

Адаптуй власні назви згідно з правилами української мови. Наприклад:
- Google → Google (без змін)
- Microsoft → Microsoft (без змін)
- OpenAI → OpenAI (без змін)

Перекладай ЛИШЕ поданий текст, нічого не додавай від себе."""

            user_prompt = f"Переклади цей текст українською мовою:\n\n{text}"
            
            # Створюємо повідомлення для API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Асинхронний виклик за допомогою to_thread
            model_name = "gpt-4o" if len(text) > 1000 else "gpt-3.5-turbo"
            
            # Готуємо функцію для запуску в потоці
            def call_openai():
                try:
                    chat_model = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model=model_name,
                        temperature=0.1
                    )
                    return chat_model.invoke(messages)
                except Exception as e:
                    self.logger.error(f"Помилка OpenAI API: {e}")
                    return None
            
            response = await asyncio.to_thread(call_openai)
            
            if response and hasattr(response, 'content'):
                # Логування токенів
                if hasattr(response, 'usage') and response.usage:
                    log_token_usage({
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'model': model_name,
                        'total_tokens': response.usage.total_tokens
                    })
                
                log_agent_action("TranslatorAgent", "translate_with_llm_async", "Переклад успішно виконано", "completed")
                return response.content
            else:
                self.logger.warning("Не вдалося отримати відповідь від OpenAI API")
                return text
        except Exception as e:
            self.logger.error(f"Помилка при перекладі через LLM: {e}")
            log_agent_action("TranslatorAgent", "translate_with_llm_async", f"Помилка: {e}", "error")
        return text

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Аналіз тональності тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 20:
            return {"label": "нейтральна", "score": 0.0}
            
        # Обмеження довжини тексту
        max_text_length = 8000
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        # Промпт для аналізу тональності
        prompt = f"""Проаналізуй тональність наведеного тексту українською мовою.
        
        Визнач, чи є текст переважно позитивним, негативним чи нейтральним. 
        Поверни результат у форматі JSON з полями:
        - label: "позитивна", "негативна" або "нейтральна"
        - score: число від 0.0 до 1.0, що показує впевненість у класифікації
        
        Текст для аналізу:
        {text}
        
        JSON-відповідь:
        """
        
        try:
            response = self.chat_model.invoke(prompt)
            
            # Обробка відповіді для вилучення лише JSON
            import re
            json_patterns = [
                r'\{.*\}',
                r'```json\s*([\s\S]*?)```',
                r'```\s*([\s\S]*?)```',
            ]
            
            json_content = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response.content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0).strip()
                    if pattern.startswith('```'):
                        json_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_content, flags=re.DOTALL)
                    break
            
            if not json_content:
                json_content = response.content
            
            try:
                result = json.loads(json_content)
                return result
            except json.JSONDecodeError:
                # Якщо не вдалося розпарсити JSON, повертаємо значення за замовчуванням
                return {"label": "нейтральна", "score": 0.0}
            
        except Exception as e:
            self.logger.error(f"Помилка аналізу тональності: {e}")
            return {"label": "нейтральна", "score": 0.0}

    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Переклад списку статей"""
        log_agent_action("TranslatorAgent", "run", f"Переклад {len(articles)} статей")
        start_time = time.time()
        
        for article in articles:
            # Переклад заголовка
            title = article.get('title', '')
            if title:
                article['title'] = self.translate_with_llm(title)
            
            # Переклад опису
            summary = article.get('summary', '')
            if summary:
                article['summary'] = self.translate_with_llm(summary)
        
        end_time = time.time()
        self.logger.info(f"Переклад {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронний переклад списку статей"""
        log_agent_action("TranslatorAgent", "arun", f"Асинхронний переклад {len(articles)} статей")
        start_time = time.time()
        
        tasks = []
        for article in articles:
            # Підготовка завдань для перекладу заголовків
            title = article.get('title', '')
            if title:
                tasks.append(('title', self.translate_with_llm_async(title), article))
            
            # Підготовка завдань для перекладу описів
            summary = article.get('summary', '')
            if summary:
                tasks.append(('summary', self.translate_with_llm_async(summary), article))
        
        # Очікуємо завершення всіх завдань
        for field, task, article in tasks:
            result = await task
            article[field] = result
        
        end_time = time.time()
        self.logger.info(f"Асинхронний переклад {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles

# --- SentimentAnalysisAgent ---
class SentimentAnalysisAgent(Agent):
    def __init__(self):
        self.logger = get_logger("sentiment_agent")
        
        # OpenAI API для аналізу тональності
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
        
        self.logger.info("SentimentAnalysisAgent ініціалізовано")

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Аналіз тональності тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 20:
            return {"label": "нейтральна", "score": 0.0}
            
        # Обмеження довжини тексту
        max_text_length = 8000
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        # Промпт для аналізу тональності
        prompt = f"""Проаналізуй тональність наведеного тексту українською мовою.
        
        Визнач, чи є текст переважно позитивним, негативним чи нейтральним. 
        Поверни результат у форматі JSON з полями:
        - label: "позитивна", "негативна" або "нейтральна"
        - score: число від 0.0 до 1.0, що показує впевненість у класифікації
        
        Текст для аналізу:
        {text}
        
        JSON-відповідь:
        """
        
        try:
            response = self.chat_model.invoke(prompt)
            
            # Обробка відповіді для вилучення лише JSON
            import re
            json_patterns = [
                r'\{.*\}',
                r'```json\s*([\s\S]*?)```',
                r'```\s*([\s\S]*?)```',
            ]
            
            json_content = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response.content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0).strip()
                    if pattern.startswith('```'):
                        json_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_content, flags=re.DOTALL)
                    break
            
            if not json_content:
                json_content = response.content
            
            try:
                result = json.loads(json_content)
                return result
            except json.JSONDecodeError:
                # Якщо не вдалося розпарсити JSON, повертаємо значення за замовчуванням
                return {"label": "нейтральна", "score": 0.0}
            
        except Exception as e:
            self.logger.error(f"Помилка аналізу тональності: {e}")
            return {"label": "нейтральна", "score": 0.0}
            
    async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
        """Асинхронний аналіз тональності тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 20:
            return {"label": "нейтральна", "score": 0.0}
            
        # Обмеження довжини тексту
        max_text_length = 8000
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        # Промпт для аналізу тональності
        prompt = f"""Проаналізуй тональність наведеного тексту українською мовою.
        
        Визнач, чи є текст переважно позитивним, негативним чи нейтральним. 
        Поверни результат у форматі JSON з полями:
        - label: "позитивна", "негативна" або "нейтральна"
        - score: число від 0.0 до 1.0, що показує впевненість у класифікації
        
        Текст для аналізу:
        {text}
        
        JSON-відповідь:
        """
        
        try:
            # Використовуємо to_thread для запуску синхронної функції в окремому потоці
            response = await asyncio.to_thread(self.chat_model.invoke, prompt)
            
            # Обробка відповіді для вилучення лише JSON
            import re
            json_patterns = [
                r'\{.*\}',
                r'```json\s*([\s\S]*?)```',
                r'```\s*([\s\S]*?)```',
            ]
            
            json_content = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response.content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0).strip()
                    if pattern.startswith('```'):
                        json_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_content, flags=re.DOTALL)
                    break
            
            if not json_content:
                json_content = response.content
            
            try:
                result = json.loads(json_content)
                return result
            except json.JSONDecodeError:
                # Якщо не вдалося розпарсити JSON, повертаємо значення за замовчуванням
                return {"label": "нейтральна", "score": 0.0}
            
        except Exception as e:
            self.logger.error(f"Помилка асинхронного аналізу тональності: {e}")
            return {"label": "нейтральна", "score": 0.0}

    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Аналіз тональності для списку статей"""
        log_agent_action("SentimentAnalysisAgent", "run", f"Аналіз тональності для {len(articles)} статей")
        start_time = time.time()
        
        for article in articles:
            # Використовуємо опис статті або заголовок для аналізу
            text = article.get('summary', '') or article.get('title', '')
            if text:
                sentiment_data = self.analyze_sentiment(text)
                article['sentiment'] = sentiment_data.get('label', 'нейтральна')
                article['sentiment_score'] = sentiment_data.get('score', 0.0)
        
        end_time = time.time()
        self.logger.info(f"Аналіз тональності для {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронний аналіз тональності для списку статей"""
        log_agent_action("SentimentAnalysisAgent", "arun", f"Асинхронний аналіз тональності для {len(articles)} статей")
        start_time = time.time()
        
        tasks = []
        for i, article in enumerate(articles):
            # Використовуємо опис статті або заголовок для аналізу
            text = article.get('summary', '') or article.get('title', '')
            if text:
                tasks.append(self.analyze_sentiment_async(text))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Порожній таск
                
        # Запускаємо всі завдання одночасно
        results = await asyncio.gather(*tasks)
        
        # Оновлюємо статті з результатами
        for i, article in enumerate(articles):
            if i < len(results) and results[i]:
                article['sentiment'] = results[i].get('label', 'нейтральна')
                article['sentiment_score'] = results[i].get('score', 0.0)
            else:
                article['sentiment'] = 'нейтральна'
                article['sentiment_score'] = 0.0
        
        end_time = time.time()
        self.logger.info(f"Асинхронний аналіз тональності для {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles

# --- SummarizerAgent ---
class SummarizerAgent(Agent):
    def __init__(self):
        self.logger = get_logger("summarizer_agent")
        
        # OpenAI API для узагальнення
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.3
            )
        
        self.logger.info("SummarizerAgent ініціалізовано")

    def summarize(self, text: str, max_length: int = 200) -> str:
        """Узагальнення тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 30:
            return ""
            
        # Обмеження довжини вхідного тексту
        max_input_length = 15000
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        # Покращений промпт для узагальнення
        prompt = f"""Створи стислий дайджест наступного тексту.
        
        Правила:
        1. Узагальнення має бути інформативним і зберігати ключові факти, числа та важливі деталі
        2. Довжина узагальнення: приблизно {max_length} символів (3-4 речення)
        3. Уникай загальних фраз типу "у статті розповідається...", просто передай ключову інформацію
        4. Використовуй об'єктивний, нейтральний тон
        5. Зберігай важливі технічні терміни
        
        Текст для узагальнення:
        {text}
        
        Дайджест:
        """
        
        try:
            response = self.chat_model.invoke(prompt)
            
            # Логування використання токенів
            if hasattr(response, 'usage') and response.usage:
                log_token_usage({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'model': self.chat_model.model_name,
                    'total_tokens': response.usage.total_tokens
                })
                
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Помилка при узагальненні тексту: {e}")
            return ""
    
    async def summarize_async(self, text: str, max_length: int = 200) -> str:
        """Асинхронне узагальнення тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 30:
            return ""
            
        # Обмеження довжини вхідного тексту
        max_input_length = 15000
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        # Покращений промпт для узагальнення
        system_prompt = """Ти експерт з узагальнення тексту українською мовою. 
Твоє завдання - створювати чіткі, інформативні та об'єктивні узагальнення, які зберігають 
ключову інформацію, факти, дані та важливі деталі з оригінального тексту.

Правила узагальнення:
1. Зберігай ключові факти, числа, імена, дати та важливі деталі
2. Уникай загальних фраз на кшталт "у статті розповідається..."
3. Використовуй об'єктивний, нейтральний тон
4. Пріоритизуй найважливішу інформацію
5. Зберігай технічні терміни без спрощення

Надай узагальнення у вигляді кількох коротких, але змістовних речень."""

        user_prompt = f"Узагальни цей текст у приблизно {max_length} символів (3-4 речення):\n\n{text}"
        
        # Створюємо повідомлення для API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Визначаємо, яку модель використовувати залежно від розміру тексту
            model_name = "gpt-4o" if len(text) > 5000 else "gpt-3.5-turbo"
            
            # Функція для виклику в окремому потоці
            def call_openai():
                try:
                    model = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model=model_name,
                        temperature=0.3
                    )
                    return model.invoke(messages)
                except Exception as e:
                    self.logger.error(f"Помилка OpenAI API: {e}")
                    return None
            
            # Запуск синхронної функції в окремому потоці
            response = await asyncio.to_thread(call_openai)
            
            if response and hasattr(response, 'content'):
                # Логування використання токенів
                if hasattr(response, 'usage') and response.usage:
                    log_token_usage({
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'model': model_name,
                        'total_tokens': response.usage.total_tokens
                    })
                    
                return response.content.strip()
            else:
                self.logger.warning("Не вдалося отримати відповідь від OpenAI API")
                return ""
                
        except Exception as e:
            self.logger.error(f"Помилка при асинхронному узагальненні тексту: {e}")
            return ""
    
    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Узагальнення статей"""
        log_agent_action("SummarizerAgent", "run", f"Узагальнення {len(articles)} статей")
        start_time = time.time()
        
        for article in articles:
            # Використовуємо повний контент або опис статті для узагальнення
            text = article.get('content', '') or article.get('summary', '')
            if text:
                ai_summary = self.summarize(text)
                if ai_summary:
                    article['ai_summary'] = ai_summary
        
        end_time = time.time()
        self.logger.info(f"Узагальнення {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронне узагальнення статей"""
        log_agent_action("SummarizerAgent", "arun", f"Асинхронне узагальнення {len(articles)} статей")
        start_time = time.time()
        
        tasks = []
        article_indices = []
        
        for i, article in enumerate(articles):
            # Використовуємо повний контент або опис статті для узагальнення
            text = article.get('content', '') or article.get('summary', '')
            if text:
                tasks.append(self.summarize_async(text))
                article_indices.append(i)
        
        # Запускаємо всі завдання одночасно
        if tasks:
            summaries = await asyncio.gather(*tasks)
            
            # Оновлюємо статті з результатами
            for i, summary in enumerate(summaries):
                if summary:
                    articles[article_indices[i]]['ai_summary'] = summary
        
        end_time = time.time()
        self.logger.info(f"Асинхронне узагальнення {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles

# --- CategorizerAgent ---
class CategorizerAgent(Agent):
    def __init__(self, categories: Dict[str, List[str]] = None):
        self.logger = get_logger("categorizer_agent")
        
        # Категорії за замовчуванням
        self.categories = categories or {
            'AI та машинне навчання': ['AI', 'штучний інтелект', 'machine learning', 'artificial intelligence', 'GPT', 'LLM', 'OpenAI', 'neural', 'нейронні мережі'],
            'Data Science': ['data', 'аналітика', 'big data', 'analytics', 'data science', 'дані'],
            'Blockchain': ['blockchain', 'блокчейн', 'crypto', 'bitcoin', 'ethereum', 'криптовалюта', 'NFT'],
            'Cloud': ['cloud', 'хмарні', 'AWS', 'Azure', 'Google Cloud', 'serverless'],
            'Кібербезпека': ['security', 'безпека', 'кібербезпека', 'hacker', 'хакер', 'vulnerability', 'вразливість']
        }
        
        # OpenAI API для категоризації
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
        
        self.logger.info(f"CategorizerAgent ініціалізовано з {len(self.categories)} категоріями")

    def categorize_by_keywords(self, title: str, summary: str) -> str:
        """Проста категоризація на основі ключових слів"""
        combined_text = (title + " " + summary).lower()
        
        for category, keywords in self.categories.items():
            if any(keyword.lower() in combined_text for keyword in keywords):
                return category
        
        return 'Інше'
    
    def categorize_with_llm(self, title: str, summary: str) -> str:
        """Категоризація з використанням LLM"""
        if not self.openai_api_key:
            return self.categorize_by_keywords(title, summary)
        
        # Підготовка категорій для промпту
        categories_text = ", ".join(self.categories.keys())
        
        prompt = f"""Визнач найбільш підходящу категорію для наступної статті. 
        Обери ЛИШЕ ОДНУ категорію із запропонованих: {categories_text}.
        Якщо стаття не підходить до жодної з категорій, вкажи 'Інше'.
        
        Стаття:
        Заголовок: {title}
        Опис: {summary}
        
        Категорія (лише назва):
        """
        
        try:
            response = self.chat_model.invoke(prompt)
            result = response.content.strip()
            
            # Перевіряємо, чи відповідь є валідною категорією
            for category in self.categories.keys():
                if category.lower() in result.lower():
                    return category
            
            return 'Інше'
        except Exception as e:
            self.logger.error(f"Помилка категоризації через LLM: {e}")
            return self.categorize_by_keywords(title, summary)
    
    async def categorize_with_llm_async(self, title: str, summary: str) -> str:
        """Асинхронна категоризація з використанням LLM"""
        if not self.openai_api_key:
            return self.categorize_by_keywords(title, summary)
        
        # Підготовка категорій для промпту
        categories_text = ", ".join(self.categories.keys())
        
        system_prompt = f"""Ти експерт з категоризації технологічних новин.
Твоє завдання - визначити точну категорію для статті на основі її заголовка та опису.

Доступні категорії:
{categories_text}

Якщо стаття не відповідає жодній з перелічених категорій, класифікуй її як 'Інше'.

Видай відповідь у форматі ТІЛЬКИ ОДНІЄЇ КАТЕГОРІЇ, без додаткових пояснень."""

        user_prompt = f"""Визнач найбільш підходящу категорію для цієї статті:
Заголовок: {title}
Опис: {summary}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Функція для виклику в окремому потоці
            def call_openai():
                try:
                    model = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model="gpt-3.5-turbo",
                        temperature=0.1
                    )
                    return model.invoke(messages)
                except Exception as e:
                    self.logger.error(f"Помилка OpenAI API: {e}")
                    return None
            
            # Запуск в окремому потоці
            response = await asyncio.to_thread(call_openai)
            
            if response and hasattr(response, 'content'):
                result = response.content.strip()
                
                # Перевіряємо, чи відповідь є валідною категорією
                for category in self.categories.keys():
                    if category.lower() in result.lower():
                        return category
                
                return 'Інше'
            else:
                self.logger.warning("Не вдалося отримати відповідь від OpenAI API")
                return self.categorize_by_keywords(title, summary)
                
        except Exception as e:
            self.logger.error(f"Помилка асинхронної категоризації: {e}")
            return self.categorize_by_keywords(title, summary)
    
    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Категоризація списку статей"""
        log_agent_action("CategorizerAgent", "run", f"Категоризація {len(articles)} статей")
        start_time = time.time()
        
        for article in articles:
            # Якщо категорія вже встановлена, пропускаємо
            if article.get('category') and article.get('category') != 'Інше':
                continue
                
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Категоризуємо статтю
            category = self.categorize_with_llm(title, summary)
            article['category'] = category
        
        end_time = time.time()
        self.logger.info(f"Категоризація {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронна категоризація списку статей"""
        log_agent_action("CategorizerAgent", "arun", f"Асинхронна категоризація {len(articles)} статей")
        start_time = time.time()
        
        tasks = []
        article_indices = []
        
        for i, article in enumerate(articles):
            # Якщо категорія вже встановлена, пропускаємо
            if article.get('category') and article.get('category') != 'Інше':
                continue
                
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Додаємо завдання для категоризації
            tasks.append(self.categorize_with_llm_async(title, summary))
            article_indices.append(i)
        
        # Запускаємо всі завдання одночасно
        if tasks:
            categories = await asyncio.gather(*tasks)
            
            # Оновлюємо статті з результатами
            for i, category in enumerate(categories):
                articles[article_indices[i]]['category'] = category
        
        end_time = time.time()
        self.logger.info(f"Асинхронна категоризація {len(articles)} статей виконано за {end_time - start_time:.2f} секунд")
        
        return articles

# --- TavilyAgent ---
class TavilyAgent(Agent):
    def __init__(self, api_key: str = None):
        """Ініціалізація агента для пошуку через Tavily AI"""
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.logger = get_logger("tavily_agent")
        
        if not self.api_key:
            self.logger.warning("Tavily API ключ не знайдено. Перевірте наявність TAVILY_API_KEY в середовищі.")
            
        self.client = None
        if self.api_key:
            self.client = TavilyClient(api_key=self.api_key)
            self.logger.info("TavilyAgent ініціалізовано успішно")
    
    def search_news(self, query: str, max_results: int = 8, include_domains: List[str] = None) -> List[Dict[str, Any]]:
        """Пошук новин за запитом через Tavily"""
        if not self.client:
            self.logger.error("Tavily API не ініціалізовано")
            return []
            
        log_agent_action("TavilyAgent", "search_news", f"Пошук новин за запитом: {query}")
        
        try:
            # Підготовка параметрів пошуку
            search_params = {
                "search_depth": "advanced",
                "max_results": max_results
            }
            
            # Якщо вказані конкретні домени, додаємо їх
            if include_domains:
                search_params["include_domains"] = ",".join(include_domains)
                
            # Виконуємо пошук
            self.logger.info(f"Виконую пошук за запитом '{query}' через Tavily API")
            response = self.client.search(query, **search_params)
            
            results = []
            for item in response.get('results', []):
                try:
                    # Форматуємо результати у вигляд статей
                    pub_date = datetime.now().isoformat()
                    
                    # Спроба отримати дату публікації з URL або заголовка
                    # Шукаємо дати у форматі YYYY/MM/DD або YYYY-MM-DD в URL
                    date_patterns = [
                        r"(\d{4})[/-](\d{2})[/-](\d{2})",  # YYYY-MM-DD або YYYY/MM/DD
                        r"(\d{2})[/-](\d{2})[/-](\d{4})"   # DD-MM-YYYY або DD/MM/YYYY
                    ]
                    
                    for pattern in date_patterns:
                        url_date_match = re.search(pattern, item.get('url', ''))
                        if url_date_match:
                            date_components = url_date_match.groups()
                            if len(date_components[0]) == 4:  # YYYY-MM-DD
                                year, month, day = date_components
                            else:  # DD-MM-YYYY
                                day, month, year = date_components
                            
                            try:
                                extracted_date = datetime(int(year), int(month), int(day))
                                if extracted_date <= datetime.now():  # Перевіряємо, що дата не в майбутньому
                                    pub_date = extracted_date.isoformat()
                                    break
                            except ValueError:
                                pass
                    
                    # Обробляємо вміст
                    content = item.get('content', '')
                    summary = content[:800] + "..." if len(content) > 800 else content
                    
                    # Створюємо об'єкт статті
                    article = {
                        'title': item.get('title', 'Без заголовка'),
                        'link': item.get('url', ''),
                        'summary': summary,
                        'content': content,
                        'source': f"Tavily: {item.get('source', '')}",
                        'pub_date': pub_date,
                        'ai_generated': False,
                        'score': item.get('score', 0)
                    }
                    
                    results.append(article)
                    
                except Exception as item_error:
                    self.logger.error(f"Помилка при обробці результату Tavily: {item_error}")
            
            self.logger.info(f"Знайдено {len(results)} результатів через Tavily API для запиту '{query}'")
            return results
        except Exception as e:
            self.logger.error(f"Помилка при пошуку через Tavily: {e}")
            return []
            
    async def search_news_async(self, query: str, max_results: int = 8, include_domains: List[str] = None) -> List[Dict[str, Any]]:
        """Асинхронний пошук новин через Tavily"""
        # Запускаємо синхронний метод в окремому потоці
        return await asyncio.to_thread(self.search_news, query, max_results, include_domains)
    
    def run(self, queries: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Синхронний запуск пошуку за запитами"""
        if isinstance(queries, str):
            queries = [queries]
            
        log_agent_action("TavilyAgent", "run", f"Пошук за {len(queries)} запитами")
        start_time = time.time()
            
        all_results = []
        for query in queries:
            results = self.search_news(query)
            all_results.extend(results)
            
        end_time = time.time()
        log_agent_action("TavilyAgent", "run", f"Знайдено {len(all_results)} статей за {end_time - start_time:.2f} секунд", "completed")
        return all_results
    
    async def arun(self, queries: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Асинхронний запуск пошуку за запитами"""
        if isinstance(queries, str):
            queries = [queries]
            
        log_agent_action("TavilyAgent", "arun", f"Асинхронний пошук за {len(queries)} запитами")
        start_time = time.time()
        
        tasks = [self.search_news_async(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # Об'єднуємо всі результати в один список
        flat_results = [item for sublist in results for item in sublist]
        
        end_time = time.time()
        log_agent_action("TavilyAgent", "arun", f"Асинхронно знайдено {len(flat_results)} статей за {end_time - start_time:.2f} секунд", "completed")
        return flat_results

# --- StorageAgent ---
class StorageAgent(Agent):
    def __init__(self, db_path: str = None):
        """Ініціалізація агента для зберігання даних"""
        self.logger = get_logger("storage_agent")
        
        # Шлях до бази даних SQLite
        self.db_path = db_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'news_data.db')
        
        # Створення таблиці, якщо не існує
        self._init_db()
        
        self.logger.info(f"StorageAgent ініціалізовано з базою даних: {self.db_path}")
    
    def _init_db(self):
        """Ініціалізація бази даних"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Створення таблиці статей
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                link TEXT UNIQUE,
                published TEXT,
                summary TEXT,
                content TEXT,
                source TEXT,
                pub_date TEXT,
                category TEXT,
                sentiment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("База даних успішно ініціалізована")
        except Exception as e:
            self.logger.error(f"Помилка при ініціалізації бази даних: {e}")
    
    def save_article(self, article: Dict[str, Any]) -> bool:
        """Збереження статті в базу даних"""
        if not article.get('title') or not article.get('link'):
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Перевірка, чи стаття з таким посиланням вже існує
            cursor.execute("SELECT id FROM articles WHERE link = ?", (article.get('link'),))
            existing = cursor.fetchone()
            
            if existing:
                # Оновлення існуючої статті
                cursor.execute('''
                UPDATE articles 
                SET title = ?, published = ?, summary = ?, content = ?, source = ?, 
                    pub_date = ?, category = ?, sentiment = ? 
                WHERE link = ?
                ''', (
                    article.get('title', ''),
                    article.get('published', ''),
                    article.get('summary', ''),
                    article.get('content', ''),
                    article.get('source', ''),
                    article.get('pub_date', datetime.now().isoformat()),
                    article.get('category', 'Інше'),
                    article.get('sentiment', 'нейтральна'),
                    article.get('link')
                ))
                self.logger.debug(f"Оновлено статтю: {article.get('title')}")
            else:
                # Вставка нової статті
                cursor.execute('''
                INSERT INTO articles 
                (title, link, published, summary, content, source, pub_date, category, sentiment) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.get('title', ''),
                    article.get('link', ''),
                    article.get('published', ''),
                    article.get('summary', ''),
                    article.get('content', ''),
                    article.get('source', ''),
                    article.get('pub_date', datetime.now().isoformat()),
                    article.get('category', 'Інше'),
                    article.get('sentiment', 'нейтральна')
                ))
                self.logger.debug(f"Додано нову статтю: {article.get('title')}")
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Помилка при збереженні статті: {e}")
            return False
    
    def get_articles(self, limit: int = 100, category: str = None) -> List[Dict[str, Any]]:
        """Отримання статей з бази даних"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Для отримання результатів як словники
            cursor = conn.cursor()
            
            query = "SELECT * FROM articles"
            params = []
            
            if category:
                query += " WHERE category = ?"
                params.append(category)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Конвертуємо рядки в словники
            articles = []
            for row in rows:
                article = dict(row)
                articles.append(article)
            
            conn.close()
            self.logger.info(f"Отримано {len(articles)} статей з бази даних")
            return articles
        except Exception as e:
            self.logger.error(f"Помилка при отриманні статей: {e}")
            return []
    
    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Збереження списку статей"""
        log_agent_action("StorageAgent", "run", f"Збереження {len(articles)} статей")
        start_time = time.time()
        
        saved_count = 0
        for article in articles:
            if self.save_article(article):
                saved_count += 1
        
        end_time = time.time()
        self.logger.info(f"Збережено {saved_count} з {len(articles)} статей за {end_time - start_time:.2f} секунд")
        
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронне збереження списку статей"""
        log_agent_action("StorageAgent", "arun", f"Асинхронне збереження {len(articles)} статей")
        start_time = time.time()
        
        # Створюємо функцію для запуску в окремому потоці
        def save_articles_batch(batch):
            saved_count = 0
            for article in batch:
                if self.save_article(article):
                    saved_count += 1
            return saved_count
        
        # Розбиваємо статті на пакети
        batch_size = 10
        batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]
        
        # Запускаємо збереження в окремих потоках
        tasks = []
        for batch in batches:
            tasks.append(asyncio.to_thread(save_articles_batch, batch))
        
        # Очікуємо результати
        results = await asyncio.gather(*tasks)
        saved_count = sum(results)
        
        end_time = time.time()
        self.logger.info(f"Асинхронно збережено {saved_count} з {len(articles)} статей за {end_time - start_time:.2f} секунд")
        
        return articles

# --- HTMLReporterAgent ---
class HTMLReporterAgent(Agent):
    def __init__(self, output_dir: str = None):
        """Ініціалізація агента для створення HTML-звітів"""
        self.logger = get_logger("html_reporter_agent")
        
        # Директорія для збереження звітів
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"HTMLReporterAgent ініціалізовано з директорією: {self.output_dir}")
    
    def generate_html(self, articles: List[Dict[str, Any]]) -> str:
        """Генерація HTML-звіту"""
        if not articles:
            return "<html><body><h1>Немає статей для відображення</h1></body></html>"
        
        # Сортування за категоріями
        articles_by_category = {}
        for article in articles:
            category = article.get('category', 'Інше')
            if category not in articles_by_category:
                articles_by_category[category] = []
            articles_by_category[category].append(article)
        
        # Створення HTML
        html = """<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Технологічні новини</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        header {
            background-color: #007bff;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .category {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 25px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .category-title {
            color: #007bff;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .article {
            border-bottom: 1px solid #eee;
            padding: 15px 0;
            margin-bottom: 15px;
        }
        .article:last-child {
            border-bottom: none;
        }
        .article-title {
            margin-top: 0;
            color: #333;
        }
        .article-title a {
            color: #007bff;
            text-decoration: none;
        }
        .article-title a:hover {
            text-decoration: underline;
        }
        .article-meta {
            color: #666;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        .article-sentiment {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .sentiment-positive {
            background-color: #d4edda;
            color: #155724;
        }
        .sentiment-negative {
            background-color: #f8d7da;
            color: #721c24;
        }
        .sentiment-neutral {
            background-color: #e2e3e5;
            color: #383d41;
        }
        .article-summary {
            line-height: 1.6;
        }
        .timestamp {
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <header>
        <h1>Технологічні новини</h1>
        <p>Останнє оновлення: """ + datetime.now().strftime("%d.%m.%Y %H:%M") + """</p>
    </header>
"""
        
        # Додаємо статті по категоріях
        for category, category_articles in articles_by_category.items():
            html += f"""
    <section class="category">
        <h2 class="category-title">{category} ({len(category_articles)})</h2>
"""
            
            for article in category_articles:
                title = article.get('title', 'Без заголовка')
                link = article.get('link', '#')
                source = article.get('source', '').replace('WebScraper: ', '').replace('TavilyAgent: ', '')
                sentiment = article.get('sentiment', 'нейтральна')
                sentiment_class = {
                    'позитивна': 'sentiment-positive',
                    'негативна': 'sentiment-negative'
                }.get(sentiment, 'sentiment-neutral')
                
                summary = article.get('summary', '')
                if len(summary) > 300:
                    summary = summary[:300] + '...'
                
                html += f"""
        <article class="article">
            <h3 class="article-title"><a href="{link}" target="_blank">{title}</a></h3>
            <div class="article-meta">
                Джерело: {source}
                <span class="article-sentiment {sentiment_class}">{sentiment}</span>
            </div>
            <div class="article-summary">
                {summary}
            </div>
        </article>
"""
            
            html += """
    </section>
"""
        
        # Закінчення HTML
        html += """
    <div class="timestamp">
        Згенеровано за допомогою NewsAgents - системи мультиагентного збору та обробки новин
    </div>
</body>
</html>
"""
        
        return html
    
    def save_html_report(self, articles: List[Dict[str, Any]]) -> str:
        """Збереження HTML-звіту в файл"""
        if not articles:
            self.logger.warning("Немає статей для створення звіту")
            return None
        
        try:
            # Генерація HTML
            html = self.generate_html(articles)
            
            # Створення імені файлу з поточною датою
            filename = f"news_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            # Запис у файл
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            self.logger.info(f"HTML-звіт збережено у файл: {filepath}")
            
            # Створення останнього звіту
            latest_filepath = os.path.join(self.output_dir, "latest_report.html")
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            self.logger.info(f"Останній HTML-звіт оновлено: {latest_filepath}")
            
            return filepath
        except Exception as e:
            self.logger.error(f"Помилка при збереженні HTML-звіту: {e}")
            return None
    
    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Створення HTML-звіту"""
        log_agent_action("HTMLReporterAgent", "run", f"Створення HTML-звіту для {len(articles)} статей")
        start_time = time.time()
        
        filepath = self.save_html_report(articles)
        
        end_time = time.time()
        if filepath:
            self.logger.info(f"HTML-звіт створено за {end_time - start_time:.2f} секунд: {filepath}")
        else:
            self.logger.error("Не вдалося створити HTML-звіт")
        
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронне створення HTML-звіту"""
        log_agent_action("HTMLReporterAgent", "arun", f"Асинхронне створення HTML-звіту для {len(articles)} статей")
        start_time = time.time()
        
        try:
            # Запускаємо створення звіту в окремому потоці
            filepath = await asyncio.to_thread(self.save_html_report, articles)
            
            end_time = time.time()
            if filepath:
                self.logger.info(f"HTML-звіт асинхронно створено за {end_time - start_time:.2f} секунд: {filepath}")
            else:
                self.logger.error("Не вдалося асинхронно створити HTML-звіт")
        except Exception as e:
            self.logger.error(f"Помилка при асинхронному створенні HTML-звіту: {e}")
            
        return articles

# --- WebScraperAgent ---
class WebScraperAgent(Agent):
    def __init__(self, base_urls: List[str] = None):
        """Ініціалізація агента для скрапінгу сайтів"""
        self.logger = get_logger("web_scraper_agent")
        
        # Завантаження URL-адрес для скрапінгу, якщо не надано
        from ukrainian_sources import UKRAINIAN_SCRAPING_URLS
        self.base_urls = base_urls or UKRAINIAN_SCRAPING_URLS
        
        # Headers для HTTP-запитів
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        
        # API-ключі для сервісів скрапінгу
        self.scraper_a_key = os.getenv("SCRAPINGBEE_API_KEY")
        self.scraper_b_key = os.getenv("UCRAWLER_API_KEY")
        
        self.logger.info(f"WebScraperAgent ініціалізовано з {len(self.base_urls)} URL-адресами для скрапінгу")
        
        # Імпортуємо детальну інформацію про джерела
        try:
            from ukrainian_sources import ALL_UKRAINIAN_SOURCES
            self.sources_info = {source['url']: source for source in ALL_UKRAINIAN_SOURCES}
            self.logger.info(f"Завантажено інформацію про {len(self.sources_info)} українських джерел")
        except ImportError:
            self.sources_info = {}
            self.logger.warning("Не вдалося імпортувати детальну інформацію про джерела")

    def fetch_html(self, url: str) -> str:
        """Отримання HTML-сторінки"""
        log_agent_action("WebScraperAgent", "fetch_html", f"Скрапінг сторінки: {url}")
        try:
            # Спроба через ScrapingBee, якщо ключ доступний
            if self.scraper_a_key:
                try:
                    params = {
                        'api_key': self.scraper_a_key,
                        'url': url,
                        'render_js': 'true'
                    }
                    response = requests.get("https://api.scrapingbee.com/v1/", params=params, timeout=30)
                    if response.status_code == 200:
                        self.logger.info(f"Успішно отримано HTML через ScrapingBee: {url}")
                        return response.text
                except Exception as e:
                    self.logger.error(f"Помилка ScrapingBee: {e}")
            
            # Спроба через uCrawler, якщо ключ доступний
            if self.scraper_b_key:
                try:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.scraper_b_key}'
                    }
                    data = {'url': url}
                    response = requests.post("https://api.ucrawler.app/api/v1/scrape", json=data, headers=headers, timeout=30)
                    if response.status_code == 200:
                        self.logger.info(f"Успішно отримано HTML через uCrawler: {url}")
                        return response.json().get('content', '')
                except Exception as e:
                    self.logger.error(f"Помилка uCrawler: {e}")
            
            # Пряме отримання HTML-вмісту як запасний варіант
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 200:
                self.logger.info(f"Успішно отримано HTML прямим запитом: {url}")
                return response.text
            else:
                self.logger.warning(f"Не вдалося отримати HTML. Статус: {response.status_code}")
                return ""
        except Exception as e:
            self.logger.error(f"Помилка при отриманні HTML: {e}")
            return ""
    
    async def fetch_html_async(self, url: str) -> str:
        """Асинхронне отримання HTML-сторінки"""
        log_agent_action("WebScraperAgent", "fetch_html_async", f"Асинхронний скрапінг сторінки: {url}")
        try:
            # Спроба через ScrapingBee, якщо ключ доступний
            if self.scraper_a_key:
                try:
                    params = {
                        'api_key': self.scraper_a_key,
                        'url': url,
                        'render_js': 'true'
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get("https://api.scrapingbee.com/v1/", params=params, timeout=30) as response:
                            if response.status == 200:
                                self.logger.info(f"Успішно отримано HTML через ScrapingBee: {url}")
                                return await response.text()
                except Exception as e:
                    self.logger.error(f"Помилка ScrapingBee: {e}")
            
            # Спроба через uCrawler, якщо ключ доступний
            if self.scraper_b_key:
                try:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.scraper_b_key}'
                    }
                    data = {'url': url}
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post("https://api.ucrawler.app/api/v1/scrape", json=data, headers=headers, timeout=30) as response:
                            if response.status == 200:
                                json_resp = await response.json()
                                self.logger.info(f"Успішно отримано HTML через uCrawler: {url}")
                                return json_resp.get('content', '')
                except Exception as e:
                    self.logger.error(f"Помилка uCrawler: {e}")
            
            # Пряме отримання HTML-вмісту як запасний варіант
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=20) as response:
                    if response.status == 200:
                        self.logger.info(f"Успішно отримано HTML прямим запитом: {url}")
                        return await response.text()
                    else:
                        self.logger.warning(f"Не вдалося отримати HTML. Статус: {response.status}")
                        return ""
        except Exception as e:
            self.logger.error(f"Помилка при асинхронному отриманні HTML: {e}")
            return ""
    
    def find_article_links(self, html: str, base_url: str) -> List[str]:
        """Пошук посилань на статті на сторінці"""
        if not html:
            return []
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            # Отримуємо селектор статей для поточного джерела, якщо доступно
            article_selector = None
            if base_url in self.sources_info:
                article_selector = self.sources_info[base_url].get('article_selector')
            
            # Якщо є спеціальний селектор для поточного сайту, використовуємо його
            if article_selector:
                articles = soup.select(article_selector)
                for article in articles:
                    # Шукаємо посилання в знайдених елементах
                    a_tags = article.find_all('a', href=True)
                    for a in a_tags:
                        href = a['href']
                        # Повний URL або відносний шлях
                        if href.startswith('http'):
                            links.append(href)
                        else:
                            # Видаляємо параметри URL та якорі
                            clean_href = href.split('#')[0].split('?')[0]
                            # Конвертуємо у повний URL
                            full_url = f"{base_url.rstrip('/')}/{clean_href.lstrip('/')}"
                            links.append(full_url)
            else:
                # Якщо немає спеціального селектора, шукаємо всі посилання
                a_tags = soup.find_all('a', href=True)
                
                # Фільтруємо потенційні посилання на статті
                for a in a_tags:
                    href = a['href']
                    # Пропускаємо соцмережі, категорії, теги тощо
                    if any(skip in href.lower() for skip in ['/tag/', '/category/', 'facebook.com', 'twitter.com', 'instagram.com', 'telegram.me']):
                        continue
                        
                    # Перевіряємо на наявність ключових слів, що вказують на статті
                    if any(term in href.lower() for term in ['/article/', '/news/', '/post/', '/story/', '/publication/']):
                        if href.startswith('http'):
                            links.append(href)
                        else:
                            full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                            links.append(full_url)
            
            # Видаляємо дублікати та повертаємо результат
            unique_links = list(set(links))
            self.logger.info(f"Знайдено {len(unique_links)} унікальних посилань на статті для {base_url}")
            return unique_links
        except Exception as e:
            self.logger.error(f"Помилка при пошуку посилань на статті: {e}")
            return []
    
    def extract_article_content(self, html: str, url: str) -> Dict[str, Any]:
        """Вилучення вмісту статті з HTML"""
        if not html:
            return {}
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Спроба вилучити заголовок
            title = None
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.text.strip()
            
            # Спроба вилучити дату
            published = None
            date_candidates = soup.select('time, .date, .time, .datetime, [datetime], .post-date, .publish-date, meta[property="article:published_time"]')
            for date_element in date_candidates:
                if date_element.name == 'meta':
                    published = date_element.get('content')
                    break
                elif date_element.has_attr('datetime'):
                    published = date_element['datetime']
                    break
                else:
                    published = date_element.text.strip()
                    break
            
            # Спроба вилучити основний текст
            content = ""
            
            # Пошук найбільш імовірного контейнера статті
            article_containers = soup.select('article, .article, .post, .content, .entry-content, .post-content')
            
            if article_containers:
                article_container = article_containers[0]
                
                # Видаляємо непотрібні елементи
                for element in article_container.select('script, style, iframe, .social, .share, .related, .comments, .advert, .ad'):
                    element.decompose()
                
                # Отримуємо текст абзаців
                paragraphs = article_container.select('p')
                content = '\n'.join([p.text.strip() for p in paragraphs if p.text.strip()])
            
            # Якщо абзаци не знайдено, спробуємо взяти весь текст контейнера
            if not content and article_containers:
                content = article_containers[0].text.strip()
            
            # Визначення джерела
            source = url
            if url in self.sources_info:
                source = self.sources_info[url].get('name', url)
            
            # Формуємо результат
            result = {
                'title': title,
                'link': url,
                'published': published,
                'summary': content[:500] + '...' if len(content) > 500 else content,
                'content': content,
                'source': f"WebScraper: {source}",
                'pub_date': datetime.now().isoformat()
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Помилка при вилученні вмісту статті: {e}")
            return {}
    
    def run(self, urls: List[str] = None) -> List[Dict[str, Any]]:
        """Основний метод для скрапінгу новин"""
        log_agent_action("WebScraperAgent", "run", f"Скрапінг новин з {len(urls) if urls else len(self.base_urls)} джерел")
        urls_to_scrape = urls or self.base_urls
        articles = []
        
        for base_url in urls_to_scrape:
            try:
                # Отримання HTML головної сторінки
                html = self.fetch_html(base_url)
                if not html:
                    self.logger.warning(f"Не вдалося отримати HTML для {base_url}")
                    continue
                
                # Пошук посилань на статті
                article_links = self.find_article_links(html, base_url)
                if not article_links:
                    self.logger.warning(f"Не знайдено посилань на статті для {base_url}")
                    continue
                
                # Обмежуємо кількість статей для скрапінгу
                article_links = article_links[:5]  # Беремо лише перші 5 статей
                
                # Скрапінг кожної статті
                for link in article_links:
                    try:
                        article_html = self.fetch_html(link)
                        if not article_html:
                            continue
                            
                        article_data = self.extract_article_content(article_html, link)
                        if article_data and article_data.get('title') and article_data.get('content'):
                            articles.append(article_data)
                    except Exception as e:
                        self.logger.error(f"Помилка при скрапінгу статті {link}: {e}")
            except Exception as e:
                self.logger.error(f"Помилка при обробці сайту {base_url}: {e}")
        
        self.logger.info(f"Отримано {len(articles)} статей з {len(urls_to_scrape)} джерел")
        return articles
    
    async def arun(self, urls: List[str] = None) -> List[Dict[str, Any]]:
        """Асинхронний метод для скрапінгу новин"""
        log_agent_action("WebScraperAgent", "arun", f"Асинхронний скрапінг новин з {len(urls) if urls else len(self.base_urls)} джерел")
        urls_to_scrape = urls or self.base_urls
        articles = []
        
        # Обмежуємо кількість одночасних запитів для запобігання перевантаження
        semaphore = asyncio.Semaphore(5)  # Максимум 5 одночасних запитів
        
        async def process_url(url):
            async with semaphore:
                try:
                    # Отримання HTML головної сторінки
                    html = await self.fetch_html_async(url)
                    if not html:
                        self.logger.warning(f"Не вдалося отримати HTML для {url}")
                        return []
                    
                    # Пошук посилань на статті
                    article_links = self.find_article_links(html, url)
                    if not article_links:
                        self.logger.warning(f"Не знайдено посилань на статті для {url}")
                        return []
                    
                    # Обмежуємо кількість статей для скрапінгу
                    article_links = article_links[:5]  # Беремо лише перші 5 статей
                    
                    # Створюємо задачі для кожної статті
                    article_tasks = []
                    for link in article_links:
                        article_tasks.append(process_article(link))
                    
                    # Запускаємо всі задачі та збираємо результати
                    article_results = await asyncio.gather(*article_tasks, return_exceptions=True)
                    
                    # Фільтруємо результати, відкидаючи помилки
                    return [result for result in article_results if isinstance(result, dict) and not isinstance(result, Exception)]
                except Exception as e:
                    self.logger.error(f"Помилка при асинхронній обробці сайту {url}: {e}")
                    return []
        
        async def process_article(link):
            async with semaphore:
                try:
                    article_html = await self.fetch_html_async(link)
                    if not article_html:
                        return {}
                        
                    # Використовуємо to_thread для запуску синхронного методу в окремому потоці
                    article_data = await asyncio.to_thread(self.extract_article_content, article_html, link)
                    
                    if article_data and article_data.get('title') and article_data.get('content'):
                        return article_data
                    return {}
                except Exception as e:
                    self.logger.error(f"Помилка при асинхронному скрапінгу статті {link}: {e}")
                    return {}
        
        # Створюємо задачі для кожного сайту
        tasks = [process_url(url) for url in urls_to_scrape]
        
        # Запускаємо всі задачі та збираємо результати
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Об'єднуємо результати
        for result in results:
            if isinstance(result, list) and not isinstance(result, Exception):
                articles.extend(result)
        
        self.logger.info(f"Асинхронно отримано {len(articles)} статей з {len(urls_to_scrape)} джерел")
        return articles

# Створення систем новин
class AINewsSystem:
    def __init__(self, config=None):
        """Ініціалізація системи збору новин"""
        self.config = config or {}
        self.logger = get_logger("news_system")
        
        # Ініціалізація агентів
        self.crawler_agent = CrawlerAgent(self.config.get('rss_feeds', []))
        self.translator_agent = TranslatorAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.summarizer_agent = SummarizerAgent()
        self.categorizer_agent = CategorizerAgent()
        self.tavily_agent = TavilyAgent()
        self.storage_agent = StorageAgent()
        self.html_reporter_agent = HTMLReporterAgent()

    async def arun(self):
        """Асинхронний запуск системи новин"""
        log_agent_action("AINewsSystem", "arun", "Запуск системи новин в асинхронному режимі")
        
        # Збір новин з RSS та LLM
        articles = await self.crawler_agent.arun()
        
        # Переклад списку статей
        translated_articles = await self.translator_agent.arun(articles)
        
        # Аналіз тональності тексту
        sentiment_data = self.sentiment_agent.run(translated_articles)
        
        # Узагальнення тексту
        summarized_articles = await self.summarizer_agent.arun(sentiment_data)
        
        # Категоризація списку статей
        categorized_articles = await self.categorizer_agent.arun(summarized_articles)
        
        # Пошук новин через Tavily
        tavily_articles = await self.tavily_agent.arun(self.config.get('ai_queries', []))
        
        # Об'єднання результатів
        all_articles = articles + translated_articles + sentiment_data + summarized_articles + categorized_articles + tavily_articles
        
        # Збереження статей у базу даних
        await self.storage_agent.arun(all_articles)
        
        # Створення HTML-звіту
        html_report = await self.html_reporter_agent.arun(all_articles)
        
        return all_articles, html_report

    def run(self):
        """Синхронний запуск системи новин"""
        log_agent_action("AINewsSystem", "run", "Запуск системи новин в синхронному режимі")
        
        # Збір новин з RSS та LLM
        articles = self.crawler_agent.run()
        
        # Переклад списку статей
        translated_articles = self.translator_agent.run(articles)
        
        # Аналіз тональності тексту
        sentiment_data = self.sentiment_agent.run(translated_articles)
        
        # Узагальнення тексту
        summarized_articles = self.summarizer_agent.run(sentiment_data)
        
        # Категоризація списку статей
        categorized_articles = self.categorizer_agent.run(summarized_articles)
        
        # Пошук новин через Tavily
        tavily_articles = self.tavily_agent.run(self.config.get('ai_queries', []))
        
        # Об'єднання результатів
        all_articles = articles + translated_articles + sentiment_data + summarized_articles + categorized_articles + tavily_articles
        
        # Збереження статей у базу даних
        self.storage_agent.run(all_articles)
        
        # Створення HTML-звіту
        html_report = self.html_reporter_agent.run(all_articles)
        
        return all_articles, html_report

# Налагодження лічильника токенів для обчислення вартості
token_usage = {
    'completion_tokens': 0,
    'prompt_tokens': 0,
    'total_tokens': 0,
    'estimated_cost_usd': 0
}

def add_token_usage(completion_tokens=0, prompt_tokens=0, cost_per_1k_tokens=0.002):
    """Додає використання токенів до глобального лічильника"""
    global token_usage
    token_usage['completion_tokens'] += completion_tokens
    token_usage['prompt_tokens'] += prompt_tokens
    token_usage['total_tokens'] += (completion_tokens + prompt_tokens)
    # Обчислюємо приблизну вартість (базується на цінах моделі gpt-3.5-turbo)
    token_usage['estimated_cost_usd'] += (completion_tokens + prompt_tokens) / 1000 * cost_per_1k_tokens

async def main_async():
    """Асинхронна функція для запуску системи новин"""
    print("Запуск системи новин в асинхронному режимі...")
    
    # Завантаження змінних середовища
    load_dotenv()
    
    # Створення конфігурації
    config = {
        'ai_queries': [
            'Найновіші перспективні технологічні стартапи',
            'Актуальні новини про технології та глобальні інновації',
            'Успішні ІТ-компанії та стартап-проекти за останній місяць',
            'Сучасні технологічні тренди та проривні інновації'
        ],
        'rss_feeds': [
            'https://techcrunch.com/feed/',
            'https://feeds.bbci.co.uk/news/technology/rss.xml',
            'https://ain.ua/feed/'
        ]
    }
    
    # Створення системи новин
    system = AINewsSystem(config)
    
    # Запуск системи в асинхронному режимі
    articles = await system.arun()
    
    print(f"Отримано {len(articles)} статей")
    print("Асинхронний процес завершено!")
    
    # Виведення статистики використання токенів
    print_token_usage_summary()

def main():
    """Синхронна функція для запуску системи новин"""
    print("Запуск системи новин в синхронному режимі...")
    
    # Завантаження змінних середовища
    load_dotenv()
    
    # Створення конфігурації
    config = {
        'ai_queries': [
            'Найновіші перспективні технологічні стартапи',
            'Актуальні новини про технології та глобальні інновації',
            'Успішні ІТ-компанії та стартап-проекти за останній місяць',
            'Сучасні технологічні тренди та проривні інновації'
        ],
        'rss_feeds': [
            'https://techcrunch.com/feed/',
            'https://feeds.bbci.co.uk/news/technology/rss.xml',
            'https://ain.ua/feed/'
        ]
    }
    
    # Створення системи новин
    system = AINewsSystem(config)
    
    # Запуск системи в синхронному режимі
    articles, html_report = system.run()
    
    print(f"Отримано {len(articles)} статей")
    print("Синхронний процес завершено!")
    
    # Виведення статистики використання токенів
    print_token_usage_summary()
    
    # Виправлено - html_report вже містить шлях до збереженого файлу, 
    # тому нам не потрібно викликати save_html_report
    if html_report:
        print(f"HTML-звіт збережено у файл: {html_report}")
    else:
        print("Не вдалося отримати HTML-звіт")

if __name__ == "__main__":
    # Перевірка, чи запускати в асинхронному режимі
    parser = argparse.ArgumentParser(description='Система збору та обробки новин')
    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Запустити в асинхронному режимі')
    args = parser.parse_args()
    
    if args.use_async:
        # Запуск асинхронного режиму
        asyncio.run(main_async())
    else:
        # Запуск синхронного режиму
        main()
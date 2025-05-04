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
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    async def arun(self, *args, **kwargs):
        """Асинхронний запуск агента"""
        raise NotImplementedError

# --- CrawlerAgent ---
class CrawlerAgent(Agent):
    def __init__(self, rss_feeds: List[str], scraper_a_url: str = None, scraper_b_url: str = None):
        self.rss_feeds = rss_feeds
        self.logger = get_logger("crawler_agent")
        
        # Завантаження ключів API зі змінних середовища
        self.scraper_a_url = scraper_a_url or "https://api.scrapingbee.com/v1/"
        self.scraper_a_key = os.getenv("SCRAPINGBEE_API_KEY")
        
        self.scraper_b_url = scraper_b_url or "https://api.ucrawler.app/api/v1/scrape"
        self.scraper_b_key = os.getenv("UCRAWLER_API_KEY")
        
        # OpenAI API для пошуку новин
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-4o",
                temperature=0.2
            )
        
        self.logger.info(f"CrawlerAgent ініціалізовано з {len(rss_feeds)} RSS-каналами")

    def fetch_rss(self) -> List[Dict[str, Any]]:
        log_agent_action("CrawlerAgent", "fetch_rss", f"Отримання новин з {len(self.rss_feeds)} RSS-каналів")
        
        articles = []
        # Визначаємо дату тижневої давності
        week_ago = datetime.now() - timedelta(days=7)
        
        for url in self.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
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
                        
                    articles.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', None),
                        'summary': entry.get('summary', None),
                        'source': 'RSS_' + url,
                        'pub_date': pub_date  # Зберігаємо об'єкт datetime для подальшого використання
                    })
            except Exception as e:
                print(f"RSS parsing error for {url}: {e}")
        
        print(f"Знайдено {len(articles)} актуальних статей (за останній тиждень)")
        return articles
    
    async def fetch_rss_async(self) -> List[Dict[str, Any]]:
        """Асинхронне отримання новин з RSS-каналів"""
        articles = []
        # Визначаємо дату тижневої давності
        week_ago = datetime.now() - timedelta(days=7)
        
        async def process_feed(url):
            """Асинхронна обробка одного RSS-каналу"""
            feed_articles = []
            try:
                # feedparser не є асинхронним, тому запускаємо його в окремому потоці
                feed = await asyncio.to_thread(feedparser.parse, url)
                
                for entry in feed.entries:
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
                        
                    feed_articles.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', None),
                        'summary': entry.get('summary', None),
                        'source': 'RSS_' + url,
                        'pub_date': pub_date  # Зберігаємо об'єкт datetime для подальшого використання
                    })
            except Exception as e:
                print(f"RSS parsing error for {url}: {e}")
            
            return feed_articles
        
        # Створюємо завдання для кожного RSS-каналу
        tasks = [process_feed(url) for url in self.rss_feeds]
        
        # Запускаємо всі завдання паралельно
        results = await asyncio.gather(*tasks)
        
        # Об'єднуємо результати
        for feed_articles in results:
            articles.extend(feed_articles)
        
        print(f"Асинхронно знайдено {len(articles)} актуальних статей (за останній тиждень)")
        return articles

    def fetch_html(self, url: str) -> str:
        # Спроба через ScrapingBee (Сервіс A)
        try:
            if self.scraper_a_key:
                params = {
                    'api_key': self.scraper_a_key,
                    'url': url,
                    'render_js': 'true'
                }
                resp = requests.get(self.scraper_a_url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.text
        except Exception as e:
            print(f"ScrapingBee error: {e}")
            
        # Спроба через uCrawler (Сервіс B)
        try:
            if self.scraper_b_key:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.scraper_b_key}'
                }
                data = {'url': url}
                resp = requests.post(self.scraper_b_url, json=data, headers=headers, timeout=30)
                if resp.status_code == 200:
                    return resp.json().get('content', '')
        except Exception as e:
            print(f"uCrawler error: {e}")
            
        return ""
    
    async def fetch_html_async(self, url: str) -> str:
        """Асинхронне отримання HTML-вмісту сторінки"""
        # Спроба через ScrapingBee (Сервіс A)
        try:
            if self.scraper_a_key:
                params = {
                    'api_key': self.scraper_a_key,
                    'url': url,
                    'render_js': 'true'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.scraper_a_url, params=params, timeout=30) as resp:
                        if resp.status == 200:
                            return await resp.text()
        except Exception as e:
            print(f"ScrapingBee error: {e}")
            
        # Спроба через uCrawler (Сервіс B)
        try:
            if self.scraper_b_key:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.scraper_b_key}'
                }
                data = {'url': url}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.scraper_b_url, json=data, headers=headers, timeout=30) as resp:
                        if resp.status == 200:
                            json_resp = await resp.json()
                            return json_resp.get('content', '')
        except Exception as e:
            print(f"uCrawler error: {e}")
            
        # Спроба без спеціальних сервісів
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.text()
        except Exception as e:
            print(f"Direct request error: {e}")
            
        return ""
    
    async def fetch_news_via_llm_async(self, query: str = "Останні новини про штучний інтелект", limit: int = 5) -> List[Dict[str, Any]]:
        """Асинхронне отримання новин за допомогою LLM та веб-пошуку"""
        if not self.openai_api_key:
            print("OpenAI API key not found, skipping LLM news fetching")
            return []
        
        try:
            prompt = f"""
            Будь ласка, знайди останні важливі новини на тему: {query}
            
            Видай результат у вигляді JSON-масиву з об'єктами, що містять такі поля:
            - title: заголовок новини
            - link: URL-посилання на джерело
            - summary: короткий зміст новини (кілька речень)
            - source: джерело новини
            
            Формат відповіді повинен бути ЛИШЕ JSON-масивом без додаткового тексту!
            Приклад:
            [
                {{
                    "title": "Назва новини",
                    "link": "https://example.com/news1",
                    "summary": "Короткий опис новини...",
                    "source": "Example News"
                }},
                ...
            ]
            """
            
            # Використовуємо to_thread для запуску синхронного методу в окремому потоці
            response = await asyncio.to_thread(self.chat_model.invoke, prompt)
            
            try:
                # Спроба розпарсити JSON
                try:
                    if hasattr(response, 'content'):
                        result = json.loads(response.content)
                    else:
                        print(f"Content causing error: {response}")
                        result = []
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    print(f"Content causing error: {response.content if hasattr(response, 'content') else response}")
                    
                    # Спроба виправити некоректний JSON
                    try:
                        # Знаходимо все, що між квадратними дужками
                        if hasattr(response, 'content'):
                            content = response.content
                            match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                            if match:
                                fixed_json = f"[{match.group(1)}]"
                                result = json.loads(fixed_json)
                            else:
                                result = []
                        else:
                            result = []
                    except Exception as json_fix_error:
                        print("Failed to fix and parse JSON")
                        result = []
                
                # Форматуємо отримані дані
                articles = []
                
                for item in result:
                    # Отримуємо поточну дату як дату публікації
                    pub_date = datetime.now()
                    
                    articles.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', '#'),
                        'published': item.get('published', None),
                        'summary': item.get('summary', ''),
                        'source': item.get('source', 'AI Generated'),
                        'pub_date': pub_date,
                        'ai_generated': True
                    })
                
                return articles
                
            except Exception as parse_error:
                print(f"Error parsing LLM response: {parse_error}")
                return []
                
        except Exception as e:
            print(f"Error fetching news via LLM: {e}")
            return []

    def run(self, ai_query: str = None):
        """Збір новин з RSS та LLM"""
        articles = self.fetch_rss()
        print(f"Зібрано {len(articles)} статей з RSS")
        
        # Якщо заданий запит для AI, додаємо новини з LLM
        if ai_query:
            ai_articles = self.fetch_news_via_llm_async(ai_query)
            # Оскільки це асинхронний метод, викликаємо його через синхронний інтерфейс
            import asyncio
            ai_articles = asyncio.run(ai_articles)
            print(f"Зібрано {len(ai_articles)} статей через AI")
            articles.extend(ai_articles)
            
        return articles
    
    async def arun(self, ai_query: str = None):
        """Асинхронний збір новин з RSS та LLM"""
        # Запускаємо збір RSS та LLM паралельно
        tasks = [self.fetch_rss_async()]
        
        # Якщо є запит для AI, додаємо його в задачі
        if ai_query:
            tasks.append(self.fetch_news_via_llm_async(ai_query))
        
        # Запускаємо всі задачі одночасно
        results = await asyncio.gather(*tasks)
        
        # Об'єднуємо результати
        articles = results[0]  # Результати RSS
        
        if ai_query and len(results) > 1:
            ai_articles = results[1]  # Результати LLM
            print(f"Асинхронно зібрано {len(ai_articles)} статей через AI")
            articles.extend(ai_articles)
        
        print(f"Всього асинхронно зібрано {len(articles)} статей")
        return articles

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
        
        self.logger.info("TranslatorAgent ініціалізовано")
        if self.api_url and self.api_key:
            self.logger.debug("Налаштовано зовнішнє API для перекладу")
        if self.openai_api_key:
            self.logger.debug("Налаштовано переклад через OpenAI")

    def translate_with_api(self, text: str, target_lang: str = 'uk') -> str:
        log_agent_action("TranslatorAgent", "translate_with_api", f"Переклад тексту ({len(text)} символів) на {target_lang}")
        
        try:
            if not self.api_url or not self.api_key:
                self.logger.warning("API для перекладу не налаштовано")
                return text
                
            resp = requests.post(self.api_url, json={
                'q': text,
                'target': target_lang,
                'key': self.api_key
            }, timeout=10)
            
            if resp.status_code == 200:
                result = resp.json().get('translatedText', text)
                log_agent_action("TranslatorAgent", "translate_with_api", "Переклад успішно виконано", "completed")
                return result
            else:
                self.logger.warning(f"API перекладу повернуло статус {resp.status_code}")
                return text
        except Exception as e:
            self.logger.error(f"Помилка API перекладу: {e}")
            log_agent_action("TranslatorAgent", "translate_with_api", f"Помилка: {e}", "error")
        return text
    
    async def translate_with_api_async(self, text: str, target_lang: str = 'uk') -> str:
        """Асинхронний переклад через API"""
        log_agent_action("TranslatorAgent", "translate_with_api_async", f"Асинхронний переклад тексту ({len(text)} символів) на {target_lang}")
        
        try:
            if not self.api_url or not self.api_key:
                self.logger.warning("API для перекладу не налаштовано")
                return text
                
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json={
                    'q': text,
                    'target': target_lang,
                    'key': self.api_key
                }, timeout=10) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        translated_text = result.get('translatedText', text)
                        log_agent_action("TranslatorAgent", "translate_with_api_async", "Переклад успішно виконано", "completed")
                        return translated_text
                    else:
                        self.logger.warning(f"API перекладу повернуло статус {resp.status}")
                        return text
        except Exception as e:
            self.logger.error(f"Асинхронна помилка API перекладу: {e}")
            log_agent_action("TranslatorAgent", "translate_with_api_async", f"Помилка: {e}", "error")
        return text
    
    def translate_with_llm(self, text: str, target_lang: str = 'uk') -> str:
        """Переклад тексту з використанням мовної моделі"""
        if not self.llm:
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
        """Асинхронний переклад тексту за допомогою LLM"""
        log_agent_action("TranslatorAgent", "translate_with_llm_async", f"Асинхронний переклад тексту через LLM ({len(text)} символів) на {target_lang}")
        
        if not self.openai_api_key or not text or len(text.strip()) < 5:
            if not self.openai_api_key:
                self.logger.warning("Ключ OpenAI API не знайдено")
            elif not text or len(text.strip()) < 5:
                self.logger.debug("Текст занадто короткий для перекладу")
            return text
            
        try:
            prompt = f"""Переклади наступний текст українською мовою, зберігаючи оригінальний формат і стиль.
            Зроби переклад природним, уникай дослівного перекладу там, де це погіршує зрозумілість.
            Адаптуй текст так, щоб він звучав природно українською мовою.
            
            {text}
            
            Переклад:
            """
            
            # Використовуємо to_thread для запуску синхронного методу в окремому потоці
            response = await asyncio.to_thread(self.chat_model.invoke, prompt)
            
            # Логування використання токенів
            if hasattr(response, 'usage') and response.usage:
                log_token_usage({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'model': 'gpt-3.5-turbo'
                })
                self.logger.debug(f"Використання токенів: {response.usage.total_tokens} (запит: {response.usage.prompt_tokens}, відповідь: {response.usage.completion_tokens})")
            
            if response.content:
                log_agent_action("TranslatorAgent", "translate_with_llm_async", "Переклад успішно виконано", "completed")
                return response.content
            return text
            
        except Exception as e:
            self.logger.error(f"Асинхронна помилка перекладу через LLM: {e}")
            log_agent_action("TranslatorAgent", "translate_with_llm_async", f"Помилка: {e}", "error")
            return text

    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        log_agent_action("TranslatorAgent", "run", f"Переклад {len(articles)} статей")
        start_time = time.time()
        
        translated_count = 0
        for article in articles:
            # Якщо статтю згенеровано AI, вона вже може бути українською
            if article.get('ai_generated', False):
                self.logger.debug("Пропуск перекладу для AI-згенерованої статті")
                continue
                
            # Переклад заголовку
            if 'title' in article:
                article['title'] = self.translate_with_llm(article['title'])
                translated_count += 1
            
            # Переклад опису/сумарі
            if 'summary' in article and article['summary']:
                article['summary'] = self.translate_with_llm(article['summary'])
                translated_count += 1
        
        end_time = time.time()
        self.logger.info(f"Переклад виконано за {end_time - start_time:.2f} секунд")
        log_agent_action("TranslatorAgent", "run", f"Перекладено {translated_count} полів у {len(articles)} статтях", "completed")
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронний переклад статей"""
        log_agent_action("TranslatorAgent", "arun", f"Асинхронна категоризація {len(articles)} статей")
        start_time = time.time()
        
        translation_tasks = []
        
        # Створюємо список статей, які потребують перекладу
        articles_to_translate = []
        for i, article in enumerate(articles):
            # Якщо статтю згенеровано AI, вона вже може бути українською
            if article.get('ai_generated', False):
                self.logger.debug("Пропуск перекладу для AI-згенерованої статті")
                continue
                
            articles_to_translate.append((i, article))
        
        # Створюємо завдання для перекладу заголовків
        title_tasks = []
        summary_tasks = []
        
        for idx, article in articles_to_translate:
            if 'title' in article:
                title_tasks.append((idx, self.translate_with_llm_async(article['title'])))
            
            if 'summary' in article and article['summary']:
                summary_tasks.append((idx, self.translate_with_llm_async(article['summary'])))
        
        # Запускаємо паралельно переклад заголовків
        for idx, task in title_tasks:
            translated_title = await task
            articles[idx]['title'] = translated_title
        
        # Запускаємо паралельно переклад описів
        for idx, task in summary_tasks:
            translated_summary = await task
            articles[idx]['summary'] = translated_summary
        
        end_time = time.time()
        self.logger.info(f"Асинхронний переклад виконано за {end_time - start_time:.2f} секунд")
        log_agent_action("TranslatorAgent", "arun", f"Асинхронно перекладено {len(title_tasks) + len(summary_tasks)} полів у {len(articles_to_translate)} статтях", "completed")
        return articles

# --- SummarizerAgent ---
class SummarizerAgent(Agent):
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.logger = get_logger("summarizer_agent")
        
        self.text_splitter = CharacterTextSplitter(
            separator=" ",  # використовуємо пробіл як роздільник
            chunk_size=2800,
            chunk_overlap=200,
            length_function=len
        )
        
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.3
            )
            self.logger.info("SummarizerAgent ініціалізовано з OpenAI API")
        else:
            self.chat_model = None
            self.logger.warning("SummarizerAgent ініціалізовано без OpenAI API (функціональність обмежена)")
    
    def summarize(self, text: str) -> str:
        """Створення короткого опису для тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 20:
            if not self.openai_api_key:
                self.logger.warning("OpenAI API ключ не знайдено")
            elif not text:
                self.logger.debug("Порожній текст для узагальнення")
            elif len(text.strip()) < 20:
                self.logger.debug(f"Текст занадто короткий для узагальнення: {len(text.strip())} символів")
            return ""
        
        # Обмеження довжини тексту для запобігання перевищення лімітів токенів
        max_text_length = 8000
        if len(text) > max_text_length:
            self.logger.debug(f"Текст для узагальнення обмежено з {len(text)} до {max_text_length} символів")
            text = text[:max_text_length]
        
        log_agent_action("SummarizerAgent", "summarize", f"Створення короткого опису для тексту {len(text)} символів")
        
        # Промпт для створення короткого опису
        prompt = f"""Створи короткий інформативний дайджест для наступного тексту українською мовою.
        
        Вимоги до дайджесту:
        1. Виділи ключову інформацію (факти, цифри, важливі події).
        2. Обсяг: 3-4 змістовні речення, що розкривають суть.
        3. Дайджест має бути самодостатнім - читач повинен зрозуміти основну ідею без читання оригіналу.
        4. Стиль: інформативний, структурований, без оціночних суджень.
        5. Використовуй природні для української мови конструкції.
        6. Уникай канцеляризмів та бюрократичної мови.
        
        Текст для дайджесту:
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
                    'total_tokens': response.usage.total_tokens,
                    'model': 'gpt-3.5-turbo'
                })
                
            log_agent_action("SummarizerAgent", "summarize", "Створення короткого опису завершено", "completed")
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Помилка при створенні короткого опису: {e}")
            log_agent_action("SummarizerAgent", "summarize", f"Помилка: {e}", "error")
            return ""
    
    async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
        """Асинхронний аналіз тональності тексту"""
        # Створюємо корутину з блокуючим викликом
        return await asyncio.to_thread(self.analyze_sentiment, text)
    
    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Синхронний запуск агента для аналізу тональності статей"""
        for article in articles:
            # Спочатку аналізуємо заголовок, потім опис, якщо є
            text_to_analyze = article.get('title', '')
            if article.get('summary'):
                text_to_analyze += " " + article.get('summary')
            
            sentiment = self.analyze_sentiment(text_to_analyze)
            article['sentiment'] = sentiment.get('label', 'нейтральна')
            article['sentiment_score'] = sentiment.get('score', 0.0)
            
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронний запуск агента для аналізу тональності статей"""
        tasks = []
        
        for i, article in enumerate(articles):
            # Спочатку аналізуємо заголовок, потім опис, якщо є
            text_to_analyze = article.get('title', '')
            if article.get('summary'):
                text_to_analyze += " " + article.get('summary')
            
            # Створюємо задачу для асинхронного аналізу
            tasks.append(self.analyze_sentiment_async(text_to_analyze))
        
        # Чекаємо завершення всіх задач одночасно
        results = await asyncio.gather(*tasks)
        
        # Оновлюємо статті результатами аналізу
        for i, sentiment in enumerate(results):
            if i < len(articles):
                articles[i]['sentiment'] = sentiment.get('label', 'нейтральна')
                articles[i]['sentiment_score'] = sentiment.get('score', 0.0)
                articles[i]['sentiment_explanation'] = sentiment.get('explanation', '')
        
        return articles

# --- WebScraperAgent ---
class WebScraperAgent(Agent):
    def __init__(self, user_agent: str = None):
        """Ініціалізація веб-скрапера
        
        Args:
            user_agent: Рядок User-Agent для HTTP запитів
        """
        self.user_agent = user_agent or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "uk-UA,uk;q=0.8,en-US;q=0.5,en;q=0.3",
        }
        self.logger = get_logger("web_scraper_agent")
    
    def extract_article_content(self, html: str, url: str) -> Dict[str, Any]:
        """Витягує вміст статті з HTML-сторінки"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Спроба знайти заголовок
            title = None
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                title = title_tag.text.strip()
            
            # Спроба знайти основний вміст статті
            content = ""
            
            # Пошук елементів з найбільшою вагою (article, main, div.content і т.д.)
            article_candidates = []
            
            # Шукаємо стандартні теги статей
            article_tag = soup.find('article')
            if article_tag:
                article_candidates.append((article_tag, 10))
            
            main_tag = soup.find('main')
            if main_tag:
                article_candidates.append((main_tag, 8))
            
            # Шукаємо div з ключовими словами у класах або id
            for div in soup.find_all('div'):
                div_class = div.get('class', [])
                div_id = div.get('id', '')
                
                # Перетворення списку класів на рядок для пошуку
                div_class_str = ' '.join(div_class) if isinstance(div_class, list) else str(div_class)
                
                score = 0
                keywords = ['content', 'article', 'post', 'entry', 'text', 'body', 'main']
                
                for keyword in keywords:
                    if keyword in div_class_str.lower() or keyword in div_id.lower():
                        score += 5
                
                if score > 0:
                    article_candidates.append((div, score))
            
            # Сортування кандидатів за вагою (від найвищої до найнижчої)
            article_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Вибір кандидата з найбільшою вагою і текстом
            for candidate, _ in article_candidates:
                # Рахуємо кількість слів у кандидаті
                candidate_text = candidate.get_text(strip=True)
                word_count = len(candidate_text.split())
                
                # Якщо текст достатньо довгий, вибираємо цього кандидата
                if word_count > 100:
                    # Витягуємо параграфи
                    paragraphs = []
                    for p in candidate.find_all('p'):
                        if len(p.get_text(strip=True)) > 20:  # Пропускаємо короткі параграфи
                            paragraphs.append(p.get_text(strip=True))
                    
                    if paragraphs:
                        content = '\n\n'.join(paragraphs)
                    else:
                        content = candidate_text
                    break
            
            # Якщо не знайдено відповідний контейнер, спробуємо витягти всі параграфи з тіла документа
            if not content:
                paragraphs = []
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 30:  # Відфільтровуємо короткі параграфи
                        paragraphs.append(text)
                
                if paragraphs:
                    content = '\n\n'.join(paragraphs)
            
            # Отримання дати публікації
            # Спочатку шукаємо в мета-тегах
            pub_date = None
            date_meta = soup.find('meta', property='article:published_time') or soup.find('meta', {'name': 'pubdate'})
            if date_meta:
                pub_date = date_meta.get('content')
            else:
                # Шукаємо теги часу або div з класами, пов'язаними з датою
                date_tag = soup.find('time', attrs={'datetime': True})
                if date_tag:
                    pub_date = date_tag.get('datetime')
                else:
                    # Шукаємо div з ключовими словами дати в класах
                    for div in soup.find_all('div'):
                        div_class = div.get('class', [])
                        div_class_str = ' '.join(div_class) if isinstance(div_class, list) else str(div_class)
                        
                        if any(kw in div_class_str.lower() for kw in ['date', 'time', 'published']):
                            pub_date = div.get_text(strip=True)
                            break
            
            # Спробуємо знайти автора
            author = None
            author_meta = soup.find('meta', {'name': 'author'}) or soup.find('meta', property='article:author')
            if author_meta:
                author = author_meta.get('content')
            else:
                # Шукаємо теги з класами, пов'язаними з автором
                for tag in soup.find_all(['span', 'div', 'a']):
                    tag_class = tag.get('class', [])
                    tag_class_str = ' '.join(tag_class) if isinstance(tag_class, list) else str(tag_class)
                    
                    if any(kw in tag_class_str.lower() for kw in ['author', 'byline']):
                        author = tag.get_text(strip=True)
                        break
            
            return {
                'title': title or "Без заголовка",
                'content': content,
                'summary': content[:800] + "..." if len(content) > 800 else content,  # Створюємо короткий опис для відображення
                'link': url,
                'pub_date': pub_date,
                'author': author,
                'source': f"WEB_{url.split('//')[-1].split('/')[0]}"
            }
            
        except Exception as e:
            self.logger.error(f"Помилка при вилученні вмісту з {url}: {e}")
            return {
                'title': "Помилка при обробці статті",
                'content': "",
                'link': url,
                'source': f"WEB_{url.split('//')[-1].split('/')[0]}_ERROR"
            }

    def find_article_links(self, base_url: str) -> List[str]:
        """Знаходить посилання на статті на головній сторінці"""
        try:
            self.logger.info(f"Пошук посилань на статті на {base_url}")
            response = requests.get(base_url, headers=self.headers, timeout=30)
            if response.status_code != 200:
                self.logger.warning(f"Не вдалося отримати головну сторінку {base_url}: {response.status_code}")
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Отримуємо домен для формування повних URL
            domain = base_url.split('//')[1].split('/')[0]
            base_scheme = base_url.split('//')[0]
            
            # Шукаємо селектор для статей в нашому списку джерел
            article_selector = ""
            source_name = ""
            for source in ALL_SOURCES:
                if source['url'] == base_url or base_url.startswith(source['url']):
                    article_selector = source.get('article_selector', '')
                    source_name = source.get('name', '')
                    break
            
            # Словник для зберігання посилань та їх метаданних
            link_data = {}
            
            # Якщо є селектор для статей, використовуємо його
            if article_selector:
                self.logger.info(f"Використовую селектор '{article_selector}' для {source_name}")
                selectors = article_selector.split(', ')
                for selector in selectors:
                    article_elements = soup.select(selector)
                    self.logger.debug(f"Знайдено {len(article_elements)} елементів за селектором '{selector}'")
                    
                    for element in article_elements:
                        # Шукаємо дату публікації
                        date_tag = element.select_one('.date, .time, time, .published, .meta-date, [datetime], [data-datetime]')
                        date_text = None
                        date_score = 0
                        
                        if date_tag:
                            # Спроба отримати дату з атрибутів
                            date_attr = date_tag.get('datetime') or date_tag.get('data-datetime')
                            if date_attr:
                                date_text = date_attr
                                date_score = 90  # Високий пріоритет для атрибутів з датою
                            else:
                                date_text = date_tag.get_text().strip()
                                date_score = 70  # Середній пріоритет для тексту дати
                        
                        # Шукаємо посилання в елементі
                        link_tags = element.find_all('a')
                        for link in link_tags:
                            href = link.get('href')
                            if href and not href.startswith('#') and not href.startswith('javascript:'):
                                # Формуємо повний URL, якщо посилання відносне
                                if href.startswith('/'):
                                    href = f"{base_scheme}//{domain}{href}"
                                elif not (href.startswith('http://') or href.startswith('https://')):
                                    href = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                                
                                # Перевіряємо, що URL належить тому ж домену
                                if domain in href:
                                    # Витягаємо текст (можливо, заголовок)
                                    title = link.get_text().strip()
                                    
                                    # Перевірка на новинне посилання (URL-структура)
                                    url_score = 0
                                    url_path = href.split(domain, 1)[1] if domain in href else ""
                                    
                                    # Перевірка на типові шляхи новин
                                    news_paths = ['/article/', '/news/', '/post/', '/blog/', '/story/', 
                                                 '/novini/', '/statti/', '/publikacii/', '/tech/', 
                                                 '/technology/', '/tehnologii/', '/digital/']
                                    
                                    # Перевірка на дату в URL (характерно для новинних сайтів)
                                    has_date_pattern = re.search(r'/202[0-4]/\d{1,2}/', url_path) or re.search(r'/\d{4}-\d{2}-\d{2}/', url_path)
                                    
                                    if has_date_pattern:
                                        url_score += 40  # Висока ймовірність актуальної новини
                                        
                                    for path in news_paths:
                                        if path in url_path:
                                            url_score += 30
                                            break
                                    
                                    # Перевірка на наявність року/дати в URL (дуже поширено)
                                    if re.search(r'202[3-4]', url_path):
                                        url_score += 20  # Бонус за посилання з поточним/минулим роком
                                    
                                    # Виключаємо сторінки категорій, теги, сторінки про нас і т.д.
                                    if any(x in url_path for x in ['/category/', '/tag/', '/about/', '/contact/', '/page/', '/author/']):
                                        url_score -= 50
                                    
                                    # Загальна оцінка посилання
                                    total_score = url_score + date_score
                                    
                                    # Зберігаємо з оцінкою
                                    if href not in link_data or total_score > link_data[href]['score']:
                                        link_data[href] = {
                                            'url': href,
                                            'title': title,
                                            'date': date_text,
                                            'score': total_score
                                        }
            
            # Якщо не знайдено статей за селектором або їх менше 5, шукаємо додатково
            if len(link_data) < 5:
                self.logger.info(f"Недостатньо статей за селектором, шукаємо додаткові посилання")
                
                # Шукаємо всі посилання на сторінці
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        # Формуємо повний URL
                        if href.startswith('/'):
                            href = f"{base_scheme}//{domain}{href}"
                        elif not (href.startswith('http://') or href.startswith('https://')):
                            href = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                        
                        # Перевіряємо, що URL належить тому ж домену
                        if domain in href and href not in link_data:
                            url_path = href.split(domain, 1)[1] if domain in href else ""
                            
                            # Перевірка на новинне посилання
                            url_score = 0
                            
                            # Перевірка на типові шляхи та дати
                            news_paths = ['/article/', '/news/', '/post/', '/blog/', '/story/', 
                                         '/novini/', '/statti/', '/publikacii/', '/tech/', 
                                         '/technology/', '/tehnologii/', '/digital/']
                                         
                            if re.search(r'/202[3-4]/\d{1,2}/', url_path) or re.search(r'/\d{4}-\d{2}-\d{2}/', url_path):
                                url_score += 40
                                
                            for path in news_paths:
                                if path in url_path:
                                    url_score += 30
                                    break
                            
                            if re.search(r'202[3-4]', url_path):
                                url_score += 20
                            
                            if any(x in url_path for x in ['/category/', '/tag/', '/about/', '/contact/', '/page/', '/author/']):
                                url_score -= 50
                            
                            if url_score > 0:
                                link_data[href] = {
                                    'url': href,
                                    'title': link.get_text().strip(),
                                    'date': None,
                                    'score': url_score
                                }
            
            # Сортуємо посилання за оцінкою
            sorted_links = sorted(link_data.values(), key=lambda x: x['score'], reverse=True)
            
            # Перетворюємо на список URL
            result_links = [item['url'] for item in sorted_links if item['score'] > 0]
            
            # Видаляємо дублікати і обмежуємо до 10 посилань
            unique_links = list(dict.fromkeys(result_links))  # Зберігає порядок в Python 3.7+
            
            self.logger.info(f"Знайдено {len(unique_links)} унікальних посилань на статті для {base_url}")
            
            if unique_links:
                self.logger.debug(f"Найкраще посилання: {unique_links[0]}")
                
            return unique_links[:10]
            
        except Exception as e:
            self.logger.error(f"Помилка при пошуку посилань на статті для {base_url}: {e}")
            return []

    def scrape_web_page(self, url: str) -> Dict[str, Any]:
        """Синхронно скрапить веб-сторінку"""
        try:
            # Якщо URL виглядає як головна сторінка, спочатку знаходимо посилання на статті
            if url.endswith('.com') or url.endswith('.ua') or url.count('/') <= 3:
                self.logger.info(f"{url} виглядає як головна сторінка, шукаємо посилання на статті")
                article_links = self.find_article_links(url)
                
                if article_links:
                    # Скрапимо декілька статей (до 3), а не тільки першу
                    results = []
                    max_articles = min(3, len(article_links))  # Обмежуємо до 3 статей максимум
                    
                    self.logger.info(f"Знайдено {len(article_links)} посилань на статті, обробляємо перші {max_articles}")
                    
                    for i in range(max_articles):
                        article_url = article_links[i]
                        self.logger.info(f"Скрапінг статті {i+1}/{max_articles}: {article_url}")
                        
                        try:
                            response = requests.get(article_url, headers=self.headers, timeout=30)
                            
                            if response.status_code == 200:
                                result = self.extract_article_content(response.text, article_url)
                                results.append(result)
                            else:
                                self.logger.warning(f"Помилка скрапінгу {article_url}: статус-код {response.status_code}")
                        except Exception as article_error:
                            self.logger.error(f"Помилка при скрапінгу статті {article_url}: {article_error}")
                    
                    # Повертаємо тільки першу статтю, щоб зберегти сумісність з іншим кодом
                    # Результати додаткових статей будуть збережені в log-файлі для аналізу
                    if results:
                        return results[0]
            
            # Якщо це не головна сторінка або не знайдено статей, скрапимо за прямим URL
            self.logger.info(f"Скрапінг за прямим URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                html = response.text
                return self.extract_article_content(html, url)
            else:
                self.logger.warning(f"Помилка скрапінгу {url}: статус-код {response.status_code}")
                return {
                    'title': f"Помилка статус-код {response.status_code}",
                    'link': url,
                    'published': datetime.now().isoformat(),
                    'summary': f"Не вдалося отримати вміст сторінки. Код помилки: {response.status_code}",
                    'source': url.split('//')[-1].split('/')[0]
                }
        except Exception as e:
            self.logger.error(f"Помилка скрапінгу {url}: {e}")
            return {
                'title': "Помилка скрапінгу",
                'link': url,
                'published': datetime.now().isoformat(),
                'summary': f"Помилка при отриманні сторінки: {str(e)}",
                'source': url.split('//')[-1].split('/')[0]
            }
    
    async def scrape_web_page_async(self, url: str) -> Dict[str, Any]:
        """Асинхронно скрапить веб-сторінку"""
        try:
            # Якщо URL виглядає як головна сторінка, спочатку знаходимо посилання на статті (асинхронно)
            if url.endswith('.com') or url.endswith('.ua') or url.count('/') <= 3:
                self.logger.info(f"{url} виглядає як головна сторінка, шукаємо посилання на статті асинхронно")
                # Запускаємо синхронну функцію в окремому потоці
                article_links = await asyncio.to_thread(self.find_article_links, url)
                
                if article_links:
                    # Скрапимо декілька статей (до 3), а не тільки першу
                    results = []
                    max_articles = min(3, len(article_links))  # Обмежуємо до 3 статей максимум
                    
                    self.logger.info(f"Знайдено {len(article_links)} посилань на статті, асинхронно обробляємо перші {max_articles}")
                    
                    # Створюємо асинхронну сесію для всіх запитів
                    async with aiohttp.ClientSession() as session:
                        # Підготовка задач для кожної статті
                        tasks = []
                        
                        for i in range(max_articles):
                            article_url = article_links[i]
                            self.logger.info(f"Підготовка асинхронного скрапінгу статті {i+1}/{max_articles}: {article_url}")
                            
                            # Створюємо задачу для кожної статті
                            tasks.append(self._process_article_async(session, article_url, i+1, max_articles))
                        
                        # Запускаємо всі задачі паралельно
                        article_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Фільтруємо результати, виключаючи винятки
                        for result in article_results:
                            if not isinstance(result, Exception) and result:
                                results.append(result)
                        
                        # Повертаємо тільки першу статтю для сумісності
                        if results:
                            return results[0]
            
            # Якщо це не головна сторінка або не знайдено статей, скрапимо за прямим URL
            self.logger.info(f"Асинхронний скрапінг за прямим URL: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Витягнення вмісту не є асинхронним, тому запускаємо в окремому потоці
                        return await asyncio.to_thread(self.extract_article_content, html, url)
                    else:
                        self.logger.warning(f"Помилка асинхронного скрапінгу {url}: статус-код {response.status}")
                        return {
                            'title': f"Помилка статус-код {response.status}",
                            'link': url,
                            'published': datetime.now().isoformat(),
                            'summary': f"Не вдалося отримати вміст сторінки. Код помилки: {response.status}",
                            'source': url.split('//')[-1].split('/')[0]
                        }
        except Exception as e:
            self.logger.error(f"Помилка асинхронного скрапінгу {url}: {e}")
            return {
                'title': "Помилка скрапінгу",
                'link': url,
                'published': datetime.now().isoformat(),
                'summary': f"Помилка при отриманні сторінки: {str(e)}",
                'source': url.split('//')[-1].split('/')[0]
            }
    
    async def _process_article_async(self, session, article_url, idx, total):
        """Допоміжний метод для асинхронної обробки однієї статті"""
        try:
            self.logger.info(f"Асинхронний скрапінг статті {idx}/{total}: {article_url}")
            
            async with session.get(article_url, headers=self.headers, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    # Витягнення вмісту не є асинхронним, тому запускаємо в окремому потоці
                    return await asyncio.to_thread(self.extract_article_content, html, article_url)
                else:
                    self.logger.warning(f"Помилка асинхронного скрапінгу {article_url}: статус-код {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Помилка при асинхронній обробці статті {article_url}: {e}")
            return None
    
    def scrape_multiple_pages(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Синхронно скрапить декілька веб-сторінок"""
        results = []
        for url in urls:
            article = self.scrape_web_page(url)
            results.append(article)
        return results
    
    async def scrape_multiple_pages_async(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Асинхронно скрапить декілька веб-сторінок паралельно"""
        # Створюємо завдання для кожного URL
        tasks = [self._scrape_web_page_with_multiple_async(url) for url in urls]
        
        # Запускаємо всі завдання паралельно
        nested_results = await asyncio.gather(*tasks)
        
        # Об'єднуємо всі результати у плоский список
        flat_results = []
        for result_list in nested_results:
            if isinstance(result_list, list):
                flat_results.extend(result_list)
            else:
                flat_results.append(result_list)
                
        self.logger.info(f"Асинхронно скрапінуто {len(flat_results)} статей з {len(urls)} сайтів")
        return flat_results
    
    async def _scrape_web_page_with_multiple_async(self, url: str) -> List[Dict[str, Any]]:
        """Асинхронно скрапить веб-сторінку з можливістю отримання кількох статей"""
        try:
            # Якщо URL виглядає як головна сторінка, спочатку знаходимо посилання на статті
            if url.endswith('.com') or url.endswith('.ua') or url.count('/') <= 3:
                self.logger.info(f"{url} виглядає як головна сторінка, шукаємо посилання на статті")
                article_links = await asyncio.to_thread(self.find_article_links, url)
                
                if article_links:
                    # Скрапимо декілька статей
                    max_articles = min(3, len(article_links))
                    
                    self.logger.info(f"Знайдено {len(article_links)} посилань на статті, обробляємо перші {max_articles}")
                    
                    async with aiohttp.ClientSession() as session:
                        # Підготовка задач для кожної статті
                        tasks = []
                        
                        for i in range(max_articles):
                            article_url = article_links[i]
                            tasks.append(self._process_article_async(session, article_url, i+1, max_articles))
                        
                        # Запускаємо всі задачі паралельно
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Фільтруємо результати, виключаючи винятки
                        valid_results = []
                        for result in results:
                            if not isinstance(result, Exception) and result:
                                valid_results.append(result)
                        
                        if valid_results:
                            return valid_results
            
            # Якщо це не головна сторінка або не знайдено статей, скрапимо за прямим URL
            self.logger.info(f"Асинхронний скрапінг за прямим URL: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        result = await asyncio.to_thread(self.extract_article_content, html, url)
                        return [result]
                    else:
                        self.logger.warning(f"Помилка скрапінгу {url}: статус-код {response.status}")
                        return [{
                            'title': f"Помилка статус-код {response.status}",
                            'link': url,
                            'published': datetime.now().isoformat(),
                            'summary': f"Не вдалося отримати вміст сторінки. Код помилки: {response.status}",
                            'source': url.split('//')[-1].split('/')[0]
                        }]
        except Exception as e:
            self.logger.error(f"Помилка скрапінгу {url}: {e}")
            return [{
                'title': "Помилка скрапінгу",
                'link': url,
                'published': datetime.now().isoformat(),
                'summary': f"Помилка при отриманні сторінки: {str(e)}",
                'source': url.split('//')[-1].split('/')[0]
            }]
    
    def run(self, urls: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Синхронний запуск агента"""
        if isinstance(urls, str):
            if '\n' in urls or ',' in urls:
                # Розділяємо рядок на список URL
                url_list = re.split(r'[\n,]+', urls)
                url_list = [url.strip() for url in url_list if url.strip()]
            else:
                url_list = [urls]
        else:
            url_list = urls
        
        return self.scrape_multiple_pages(url_list)
    
    async def arun(self, urls: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Асинхронний запуск агента"""
        if isinstance(urls, str):
            if '\n' in urls or ',' in urls:
                # Розділяємо рядок на список URL
                url_list = re.split(r'[\n,]+', urls)
                url_list = [url.strip() for url in url_list if url.strip()]
            else:
                url_list = [urls]
        else:
            url_list = urls
        
        return await self.scrape_multiple_pages_async(url_list)

# --- CategorizerAgent ---
class CategorizerAgent(Agent):
    def __init__(self, categories: Dict[str, List[str]] = None, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.logger = get_logger("categorizer_agent")
        
        # Категорії за замовчуванням, якщо не передано власні
        self.categories = categories or {
            'AI та машинне навчання': ['AI', 'штучний інтелект', 'machine learning', 'artificial intelligence', 'GPT', 'LLM', 'OpenAI', 'neural', 'нейронні мережі'],
            'Data Science': ['data', 'аналітика', 'big data', 'analytics', 'data science', 'дані'],
            'Blockchain': ['blockchain', 'блокчейн', 'crypto', 'bitcoin', 'ethereum', 'криптовалюта', 'NFT'],
            'Cloud': ['cloud', 'хмарні', 'AWS', 'Azure', 'Google Cloud', 'serverless'],
            'Кібербезпека': ['security', 'безпека', 'кібербезпека', 'hacker', 'хакер', 'vulnerability', 'вразливість']
        }
        
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
            self.logger.info("CategorizerAgent ініціалізовано з OpenAI API")
        else:
            self.chat_model = None
            self.logger.warning("CategorizerAgent ініціалізовано без OpenAI API (функціональність обмежена)")
    
    def categorize(self, article: Dict[str, Any]) -> str:
        """Категоризація статті за її заголовком і змістом"""
        log_agent_action("CategorizerAgent", "categorize", f"Категоризація статті: {article.get('title', '')[:50]}...")
        
        if not self.openai_api_key:
            # Простий fallback на ключові слова, якщо немає API ключа
            return self._categorize_by_keywords(article)
        
        try:
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
            
            response = self.chat_model.invoke(prompt)
            
            # Логування використання токенів
            if hasattr(response, 'usage') and response.usage:
                log_token_usage({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'model': 'gpt-3.5-turbo'
                })
            
            result = response.content.strip()
            
            # Перевіряємо, чи відповідь є валідною категорією
            for category in self.categories.keys():
                if category.lower() in result.lower():
                    log_agent_action("CategorizerAgent", "categorize", f"Категорія: {category}", "completed")
                    return category
            
            # Якщо не знайдено точного співпадіння, перевіряємо за ключовими словами
            fallback_category = self._categorize_by_keywords(article)
            return fallback_category
                
        except Exception as e:
            self.logger.error(f"Помилка при категоризації: {e}")
            log_agent_action("CategorizerAgent", "categorize", f"Помилка: {e}", "error")
            return 'Інше'
    
    async def categorize_async(self, article: Dict[str, Any]) -> str:
        """Асинхронна категоризація статті"""
        log_agent_action("CategorizerAgent", "categorize_async", f"Асинхронна категоризація статті: {article.get('title', '')[:50]}...")
        
        if not self.openai_api_key:
            # Простий fallback на ключові слова, якщо немає API ключа
            return self._categorize_by_keywords(article)
        
        try:
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
            
            # Використовуємо to_thread для запуску синхронного методу в окремому потоці
            response = await asyncio.to_thread(self.chat_model.invoke, prompt)
            
            # Логування використання токенів
            if hasattr(response, 'usage') and response.usage:
                log_token_usage({
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'model': 'gpt-3.5-turbo'
                })
            
            result = response.content.strip()
            
            # Перевіряємо, чи відповідь є валідною категорією
            for category in self.categories.keys():
                if category.lower() in result.lower():
                    log_agent_action("CategorizerAgent", "categorize_async", f"Категорія: {category}", "completed")
                    return category
            
            # Якщо не знайдено точного співпадіння, перевіряємо за ключовими словами
            fallback_category = self._categorize_by_keywords(article)
            return fallback_category
                
        except Exception as e:
            self.logger.error(f"Помилка при асинхронній категоризації: {e}")
            log_agent_action("CategorizerAgent", "categorize_async", f"Помилка: {e}", "error")
            return 'Інше'
    
    def _categorize_by_keywords(self, article: Dict[str, Any]) -> str:
        """Категоризація на основі ключових слів"""
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        full_text = title + " " + summary
        
        best_category = 'Інше'
        max_matches = 0
        
        for category, keywords in self.categories.items():
            matches = sum(1 for kw in keywords if kw.lower() in full_text)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        self.logger.debug(f"Категоризація за ключовими словами: {best_category} (співпадінь: {max_matches})")
        return best_category
    
    def run(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Категоризація статей"""
        log_agent_action("CategorizerAgent", "run", f"Категоризація {len(articles)} статей")
        start_time = time.time()
        
        for article in articles:
            article['category'] = self.categorize(article)
            
        end_time = time.time()
        log_agent_action("CategorizerAgent", "run", f"Категоризовано {len(articles)} статей за {end_time - start_time:.2f} секунд", "completed")
        return articles
    
    async def arun(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Асинхронна категоризація статей"""
        log_agent_action("CategorizerAgent", "arun", f"Асинхронна категоризація {len(articles)} статей")
        start_time = time.time()
        
        # Створюємо завдання для кожної статті
        tasks = []
        for i, article in enumerate(articles):
            tasks.append((i, self.categorize_async(article)))
        
        # Паралельно виконуємо категоризацію
        for i, task in tasks:
            articles[i]['category'] = await task
            
        end_time = time.time()
        log_agent_action("CategorizerAgent", "arun", f"Асинхронно категоризовано {len(articles)} статей за {end_time - start_time:.2f} секунд", "completed")
        return articles
  
# --- AINewsSystem ---
class AINewsSystem:
    def __init__(self, config: Dict[str, Any]):
        """Ініціалізація системи новин"""
        self.config = config
        self.logger = get_logger("ainews_system")
        
        # Перевірка режиму debug
        self.debug_mode = os.getenv("DEBUG_MODE", "False").lower() in ["true", "1", "yes"]
        
        # Додаємо українські джерела для скрапінгу, якщо не вказано
        scrape_urls = config.get('scrape_urls', [])
        
        # Перевіряємо, чи є українські джерела у конфігурації
        has_ukrainian = False
        for url in scrape_urls:
            if '.ua' in url:
                has_ukrainian = True
                break
        
        # Якщо українських джерел немає, додаємо їх
        if not has_ukrainian:
            # Імпортуємо джерела з ukrainian_sources, якщо доступно
            try:
                from ukrainian_sources import UKRAINIAN_SCRAPING_URLS
                self.logger.info(f"Додавання {len(UKRAINIAN_SCRAPING_URLS)} українських джерел зі списку")
                scrape_urls.extend(UKRAINIAN_SCRAPING_URLS[:5])  # Додаємо перші 5 українських джерел
            except ImportError:
                # Якщо модуль не знайдено, додаємо декілька українських джерел вручну
                self.logger.warning("Модуль ukrainian_sources не знайдено, додаємо базові українські джерела")
                ua_sources = [
                    "https://dou.ua", 
                    "https://dev.ua", 
                    "https://ain.ua", 
                    "https://unian.ua/science", 
                    "https://24tv.ua/tech"
                ]
                scrape_urls.extend(ua_sources)
        
        # Оновлюємо конфігурацію
        self.config['scrape_urls'] = scrape_urls
        
        # Ініціалізація агентів
        self.logger.info("Ініціалізація агентів NewsAgents")
        
        # Агент для збору новин
        self.crawler_agent = CrawlerAgent(
            rss_feeds=config.get('rss_feeds', [])
        )
        
        # Агент для скрапінгу веб-сторінок
        self.web_scraper_agent = WebScraperAgent()
        
        # Агент для перекладу
        self.translator_agent = TranslatorAgent()
        
        # Агент для створення коротких описів
        self.summarizer_agent = SummarizerAgent()
        
        # Агент для аналізу тональності
        self.sentiment_agent = SentimentAnalysisAgent()
        
        # Агент для категоризації статей
        self.categorizer_agent = CategorizerAgent(
            categories=config.get('categories', {})
        )
        
        # Агент для зберігання даних
        self.storage_agent = StorageAgent(
            db_path=config.get('db_path', 'articles.db')
        )
        
        # Агент для створення HTML-звіту
        self.html_reporter_agent = HTMLReporterAgent(
            output_file=config.get('html_report_file', 'news_report.html'),
            template_file=config.get('html_template_file', None)
        )
        
        # Список платформ для публікації результатів
        self.platforms = config.get('platforms', ['console'])
        
        self.logger.info(f"AINewsSystem ініціалізовано: {len(config.get('rss_feeds', []))} RSS-каналів, " + 
                        f"{len(config.get('scrape_urls', []))} сайтів для скрапінгу, " +
                        f"{len(config.get('categories', {}))} категорій, " +
                        f"{len(self.platforms)} платформ виводу")
    
    def run(self) -> Dict[str, Any]:
        """Синхронне виконання процесу збору та обробки новин"""
        start_time = time.time()
        log_agent_action("AINewsSystem", "run", "Запуск процесу збору та обробки новин", "started")
        self.logger.info("Запуск синхронного процесу збору та обробки новин")
        
        # Крок 1: Збір новин з RSS-каналів
        self.logger.info("Крок 1: Збір новин з RSS")
        rss_articles = self.crawler_agent.run()
        
        # Крок 2: Збір новин за допомогою LLM та пошукових запитів
        ai_queries = self.config.get('ai_queries', [])
        
        llm_articles = []
        if ai_queries:
            self.logger.info(f"Крок 2: Збір новин через LLM за запитами ({len(ai_queries)} запитів)")
            for query in ai_queries:
                self.logger.debug(f"Запит LLM: {query}")
                query_articles = asyncio.run(self.crawler_agent.fetch_news_via_llm_async(query))
                llm_articles.extend(query_articles)
        
        # Крок 3: Скрапінг статей з веб-сайтів
        scrape_urls = self.config.get('scrape_urls', [])
        
        scraped_articles = []
        if scrape_urls:
            self.logger.info(f"Крок 3: Скрапінг {len(scrape_urls)} веб-сайтів")
            scraped_articles = self.web_scraper_agent.run(scrape_urls)
        
        # Об'єднання всіх статей
        all_articles = rss_articles + llm_articles + scraped_articles
        self.logger.info(f"Зібрано загалом {len(all_articles)} статей (RSS: {len(rss_articles)}, LLM: {len(llm_articles)}, Скрапінг: {len(scraped_articles)})")
        
        # Якщо статей немає, завершуємо
        if not all_articles:
            self.logger.warning("Не знайдено жодної статті. Процес зупинено.")
            log_agent_action("AINewsSystem", "run", "Не знайдено жодної статті", "completed")
            return {"status": "error", "message": "Не знайдено жодної статті"}
        
        # Крок 4: Переклад статей
        self.logger.info("Крок 4: Переклад статей")
        translated_articles = self.translator_agent.run(all_articles)
        
        # Крок 5: Створення коротких описів
        self.logger.info("Крок 5: Створення коротких описів")
        summarized_articles = self.summarizer_agent.run(translated_articles)
        
        # Крок 6: Аналіз тональності
        self.logger.info("Крок 6: Аналіз тональності")
        sentiment_articles = self.sentiment_agent.run(summarized_articles)
        
        # Крок 7: Категоризація
        self.logger.info("Крок 7: Категоризація статей")
        categorized_articles = self.categorizer_agent.run(sentiment_articles)
        
        # Крок 8: Збереження в базу даних
        self.logger.info("Крок 8: Збереження статей у базу даних")
        self.storage_agent.run(categorized_articles)
        
        # Крок 9: Формування звіту
        if 'html' in self.platforms:
            self.logger.info("Крок 9: Формування HTML-звіту")
            self.html_reporter_agent.run(categorized_articles)
        
        # Вивід на консоль
        if 'console' in self.platforms:
            self.logger.info("Вивід на консоль:")
            for article in categorized_articles[:5]:  # виводимо лише 5 перших
                self.logger.info(f"- {article['title']} (Категорія: {article.get('category', 'Невизначена')}, Тональність: {article.get('sentiment', 'невизначена')})")
                if article.get('ai_summary'):
                    self.logger.info(f"  Короткий опис: {article['ai_summary']}")
                self.logger.info("---")
        
        end_time = time.time()
        process_time = end_time - start_time
        self.logger.info(f"Процес завершено за {process_time:.2f} секунд")
        
        # Логування використання токенів
        print_token_usage_summary()
        
        log_agent_action("AINewsSystem", "run", f"Процес завершено, оброблено {len(all_articles)} статей", "completed")
        return {
            "status": "success",
            "articles_count": len(categorized_articles),
            "processing_time": process_time
        }
    
    async def arun(self) -> Dict[str, Any]:
        """Асинхронне виконання процесу збору та обробки новин"""
        start_time = time.time()
        log_agent_action("AINewsSystem", "arun", "Запуск асинхронного процесу збору та обробки новин", "started")
        self.logger.info("Запуск асинхронного процесу збору та обробки новин")
        
        # Підготовка задач для паралельного збору новин
        collection_tasks = []
        
        # Задача 1: Збір новин з RSS-каналів
        self.logger.info("Крок 1: Асинхронний збір новин з RSS")
        rss_task = self.crawler_agent.arun()
        collection_tasks.append(("rss", rss_task))
        
        # Задача 2: Збір новин за допомогою LLM та пошукових запитів
        ai_queries = self.config.get('ai_queries', [])
        
        if ai_queries:
            self.logger.info(f"Крок 2: Асинхронний збір новин через LLM за запитами ({len(ai_queries)} запитів)")
            for i, query in enumerate(ai_queries):
                self.logger.debug(f"Запит LLM {i+1}: {query}")
                llm_task = self.crawler_agent.fetch_news_via_llm_async(query)
                collection_tasks.append((f"llm_{i}", llm_task))
        
        # Задача 3: Скрапінг статей з веб-сайтів
        scrape_urls = self.config.get('scrape_urls', [])
        
        if scrape_urls:
            self.logger.info(f"Крок 3: Асинхронний скрапінг {len(scrape_urls)} веб-сайтів")
            scraper_task = self.web_scraper_agent.arun(scrape_urls)
            collection_tasks.append(("scrape", scraper_task))
        
        # Запускаємо всі задачі збору одночасно
        self.logger.debug(f"Запуск {len(collection_tasks)} паралельних задач для збору новин")
        
        # Асинхронне виконання всіх задач
        collection_results = {}
        for name, task in collection_tasks:
            try:
                result = await task
                collection_results[name] = result
            except Exception as e:
                self.logger.error(f"Помилка в задачі збору '{name}': {e}")
                collection_results[name] = []
        
        # Об'єднання всіх статей
        all_articles = []
        rss_articles = collection_results.get("rss", [])
        all_articles.extend(rss_articles)
        
        scraped_articles = collection_results.get("scrape", [])
        all_articles.extend(scraped_articles)
        
        llm_articles = []
        for key, value in collection_results.items():
            if key.startswith("llm_"):
                llm_articles.extend(value)
                all_articles.extend(value)
        
        self.logger.info(f"Асинхронно зібрано загалом {len(all_articles)} статей (RSS: {len(rss_articles)}, LLM: {len(llm_articles)}, Скрапінг: {len(scraped_articles)})")
        
        # Якщо статей немає, завершуємо
        if not all_articles:
            self.logger.warning("Не знайдено жодної статті. Процес зупинено.")
            log_agent_action("AINewsSystem", "arun", "Не знайдено жодної статті", "completed")
            return {"status": "error", "message": "Не знайдено жодної статті"}
        
        # Паралельне виконання обробки (переклад, узагальнення, аналіз тональності, категоризація)
        self.logger.info("Кроки 4-7: Паралельна обробка статей (переклад, узагальнення, тональність, категоризація)")
        
        # Крок 4: Переклад статей
        translated_articles = await self.translator_agent.arun(all_articles)
        
        # Крок 5: Створення коротких описів
        summarized_articles = await self.summarizer_agent.arun(translated_articles)
        
        # Крок 6: Аналіз тональності
        sentiment_articles = await self.sentiment_agent.arun(summarized_articles)
        
        # Крок 7: Категоризація
        categorized_articles = await self.categorizer_agent.arun(sentiment_articles)
        
        # Крок 8: Збереження в базу даних
        self.logger.info("Крок 8: Збереження статей у базу даних")
        await self.storage_agent.arun(categorized_articles)
        
        # Крок 9: Формування звіту
        if 'html' in self.platforms:
            self.logger.info("Крок 9: Формування HTML-звіту")
            await self.html_reporter_agent.arun(categorized_articles)
        
        # Вивід на консоль
        if 'console' in self.platforms:
            self.logger.info("Вивід на консоль:")
            for article in categorized_articles[:5]:  # виводимо лише 5 перших
                self.logger.info(f"- {article['title']} (Категорія: {article.get('category', 'Невизначена')}, Тональність: {article.get('sentiment', 'невизначена')})")
                if article.get('ai_summary'):
                    self.logger.info(f"  Короткий опис: {article['ai_summary']}")
                self.logger.info("---")
        
        end_time = time.time()
        process_time = end_time - start_time
        self.logger.info(f"Асинхронний процес завершено за {process_time:.2f} секунд")
        
        # Логування використання токенів
        print_token_usage_summary()
        
        log_agent_action("AINewsSystem", "arun", f"Асинхронний процес завершено, оброблено {len(all_articles)} статей", "completed")
        return {
            "status": "success",
            "articles_count": len(categorized_articles),
            "processing_time": process_time
        }

# --- Основна функція ---
if __name__ == "__main__":
    # Конфігурація джерел новин
    # Можна вибрати один з варіантів або комбінувати їх
    use_ukrainian_sources = True  # Використовувати українські джерела
    use_international_sources = True  # Використовувати міжнародні джерела
    use_rss_feeds = True  # Використовувати RSS-канали
    
    # Список URL-ів для скрапінгу в залежності від вибраних опцій
    scrape_urls = []
    if use_ukrainian_sources:
        scrape_urls.extend(UKRAINIAN_SCRAPING_URLS[:3])  # Перші 3 українських джерела
    if use_international_sources:
        scrape_urls.extend(INTERNATIONAL_SCRAPING_URLS[:3])  # Перші 3 міжнародних джерела
    
    # Основна конфігурація системи
    config = {
        'rss_feeds': [
            # Технологічні новини
            'https://feeds.bbci.co.uk/news/technology/rss.xml',
            'https://venturebeat.com/category/ai/feed/',
        ] if use_rss_feeds else [],
        
        'ai_queries': [
            'Останні новини про штучний інтелект та технології в Україні',
            'Останні IT-новини'
        ],
        
        'scrape_urls': scrape_urls,
        
        'categories': {
            'AI та машинне навчання': ['AI', 'штучний інтелект', 'machine learning', 'artificial intelligence', 'нейронні мережі', 'GPT', 'LLM'],
            'Data Science': ['data', 'аналітика', 'big data', 'дані', 'аналізу даних', 'data science'],
            'Програмування': ['код', 'розробка', 'programming', 'development', 'розробники', 'developer'],
            'Кібербезпека': ['безпека', 'security', 'кібербезпека', 'хакери', 'витік даних', 'атака'],
            'Інфраструктура': ['cloud', 'хмарні', 'дата-центр', 'сервери', 'DevOps', 'контейнери', 'kubernetes'],
            'Телекомунікації': ['мережі', 'network', 'інтернет', 'телеком', '5G', 'звязок'],
            'Технології': ['gadgets', 'гаджети', 'стартап', 'технології', 'технологічні', 'tech']
        },
        
        'platforms': ['console', 'html'],
        'db_path': 'articles.db',
        'html_report_file': 'news_report.html'
    }
    
    # Перевірка режиму запуску
    async_mode = os.getenv("ASYNC_MODE", "False").lower() in ["true", "1", "yes"]
    
    # Створення системи
    news_system = AINewsSystem(config)
    
    # Запуск системи
    if async_mode:
        logger.info("Запуск в асинхронному режимі")
        import asyncio
        result = asyncio.run(news_system.arun())
    else:
        logger.info("Запуск в синхронному режимі")
        result = news_system.run()
    
    # Вивід результатів
    print(f"Результат: {result['status']}")
    if result['status'] == 'success':
        print(f"Оброблено {result['articles_count']} статей")
        print(f"Процес виконано за {result['processing_time']:.2f} секунд")
        if 'html' in news_system.platforms:
            print(f"HTML-звіт збережено у файл: {news_system.html_reporter_agent.output_file}")
    
    # Для порівняння можна запустити синхронну версію, але вона буде працювати набагато повільніше
    use_sync_version = False  # Змініть на True, якщо хочете порівняти швидкість
    
    # Якщо було запущено в асинхронному режимі, і додатково потрібна синхронна версія
    if async_mode and use_sync_version:
        print("\nДля порівняння запускаємо синхронну версію...")
        
        start_time = time.time()
        sync_result = news_system.run()
        end_time = time.time()
        
        print(f"Синхронна версія виконана за {end_time - start_time:.2f} секунд")
        print(f"Оброблено {sync_result['articles_count']} статей")
        print(f"HTML-звіт збережено у файл: {news_system.html_reporter_agent.output_file}")

# --- SentimentAnalysisAgent ---
class SentimentAnalysisAgent(Agent):
    def __init__(self, openai_api_key: str = None):
        """Ініціалізація агента для аналізу тональності"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.logger = get_logger("sentiment_agent")
        
        if self.openai_api_key:
            self.chat_model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.1
            )
        else:
            self.logger.warning("API ключ OpenAI не налаштовано")
            self.chat_model = None

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Аналіз тональності тексту"""
        if not self.openai_api_key or not text or len(text.strip()) < 20:
            return {"label": "нейтральна", "score": 0.0}
        
        log_agent_action("SentimentAnalysisAgent", "analyze_sentiment", f"Аналіз тональності тексту ({len(text)} символів)")
        
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
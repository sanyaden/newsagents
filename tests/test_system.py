import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import asyncio
import tempfile
import sqlite3
from datetime import datetime

# Додаємо каталог батьківського проекту до шляху пошуку модулів
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Імпортуємо класи для тестування
from main import (
    CrawlerAgent, 
    WebScraperAgent, 
    TranslatorAgent, 
    SummarizerAgent, 
    SentimentAnalysisAgent, 
    StorageAgent
)

from main import AINewsSystem

class TestAINewsSystem(unittest.TestCase):
    """Системні тести для повного процесу обробки новин"""
    
    def setUp(self):
        """Підготовка до тестів"""
        # Створюємо тимчасову базу даних для тестів
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Створюємо тимчасовий HTML-звіт
        self.temp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        self.temp_html.close()
        
        # Тестова конфігурація
        self.config = {
            'rss_feeds': [
                'https://feeds.bbci.co.uk/news/technology/rss.xml'
            ],
            'categories': {
                'AI та машинне навчання': ['AI', 'штучний інтелект', 'machine learning'],
                'Data Science': ['data', 'аналітика', 'big data']
            },
            'platforms': ['console'],
            'ai_queries': [
                'Останні новини про штучний інтелект'
            ],
            'scrape_urls': [
                'https://example.com/test'
            ],
            'db_path': self.temp_db.name,
            'html_report_file': self.temp_html.name
        }
        
        # Створюємо систему
        self.news_system = AINewsSystem(self.config)
        
        # Створюємо тестові дані
        self.test_articles = [
            {
                'title': 'Test AI News',
                'link': 'https://example.com/ai-news',
                'published': 'Fri, 28 Jun 2024 10:00:00 GMT',
                'summary': 'This is a test news article about artificial intelligence.',
                'source': 'RSS_test'
            },
            {
                'title': 'Test Data Science News',
                'link': 'https://example.com/data-science',
                'published': 'Fri, 28 Jun 2024 11:00:00 GMT',
                'summary': 'This is a test news article about data science and analytics.',
                'source': 'RSS_test'
            }
        ]
    
    def tearDown(self):
        """Очищення після тестів"""
        # Видаляємо тимчасові файли
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
        if os.path.exists(self.temp_html.name):
            os.unlink(self.temp_html.name)
    
    @patch.object(CrawlerAgent, 'fetch_rss')
    @patch.object(WebScraperAgent, 'run')
    @patch.object(TranslatorAgent, 'run')
    @patch.object(SummarizerAgent, 'run')
    @patch.object(SentimentAnalysisAgent, 'run')
    def test_system_sync_flow(self, mock_sentiment, mock_summarizer, 
                             mock_translator, mock_scraper, mock_crawler):
        """Тестування синхронного потоку обробки новин"""
        # Налаштування моків
        mock_crawler.return_value = self.test_articles
        mock_scraper.return_value = []  # Порожній список статей зі скрапера для простоти
        
        # Функції для імітації процесу обробки
        def translator_side_effect(articles):
            # Імітація перекладу
            for article in articles:
                if 'artificial intelligence' in article.get('summary', ''):
                    article['title'] = 'Тестові новини про ШІ'
                    article['summary'] = 'Це тестова новина про штучний інтелект.'
                elif 'data science' in article.get('summary', ''):
                    article['title'] = 'Тестові новини про науку про дані'
                    article['summary'] = 'Це тестова новина про науку про дані та аналітику.'
            return articles
            
        def summarizer_side_effect(articles):
            # Імітація узагальнення
            for article in articles:
                if 'штучний інтелект' in article.get('summary', ''):
                    article['ai_summary'] = 'Короткий огляд новин про ШІ.'
                elif 'науку про дані' in article.get('summary', ''):
                    article['ai_summary'] = 'Короткий огляд новин про аналітику даних.'
            return articles
            
        def sentiment_side_effect(articles):
            # Імітація аналізу тональності
            for article in articles:
                article['sentiment'] = 'нейтральна'
                article['sentiment_score'] = 0.0
            return articles
        
        # Встановлюємо side effects для моків
        mock_translator.side_effect = translator_side_effect
        mock_summarizer.side_effect = summarizer_side_effect
        mock_sentiment.side_effect = sentiment_side_effect
        
        # Запускаємо систему, але мокуємо виклик методів зберігання
        with patch.object(StorageAgent, 'run', return_value=None):
            with patch('main.HTMLReporterAgent') as mock_html:
                mock_html.return_value.run.return_value = None
                
                result = self.news_system.run()
        
        # Перевірка, що всі агенти були викликані
        mock_crawler.assert_called_once()
        mock_translator.assert_called_once()
        mock_summarizer.assert_called_once()
        mock_sentiment.assert_called_once()
    
    @patch.object(CrawlerAgent, 'arun')
    @patch.object(WebScraperAgent, 'arun')
    @patch.object(TranslatorAgent, 'arun')
    @patch.object(SummarizerAgent, 'arun')
    @patch.object(SentimentAnalysisAgent, 'arun')
    @patch.object(StorageAgent, 'arun')
    def test_system_async_flow(self, mock_storage, mock_sentiment, 
                              mock_summarizer, mock_translator, 
                              mock_scraper, mock_crawler):
        """Тестування асинхронного потоку обробки новин"""
        # Налаштування моків для асинхронних методів
        async def crawler_async():
            return self.test_articles
            
        async def scraper_async(urls):
            return []
            
        async def translator_async(articles):
            # Імітація асинхронного перекладу
            for article in articles:
                if 'artificial intelligence' in article.get('summary', ''):
                    article['title'] = 'Тестові новини про ШІ'
                    article['summary'] = 'Це тестова новина про штучний інтелект.'
                elif 'data science' in article.get('summary', ''):
                    article['title'] = 'Тестові новини про науку про дані'
                    article['summary'] = 'Це тестова новина про науку про дані та аналітику.'
            return articles
            
        async def summarizer_async(articles):
            # Імітація асинхронного узагальнення
            for article in articles:
                if 'штучний інтелект' in article.get('summary', ''):
                    article['ai_summary'] = 'Короткий огляд новин про ШІ.'
                elif 'науку про дані' in article.get('summary', ''):
                    article['ai_summary'] = 'Короткий огляд новин про аналітику даних.'
            return articles
            
        async def sentiment_async(articles):
            # Імітація асинхронного аналізу тональності
            for article in articles:
                article['sentiment'] = 'нейтральна'
                article['sentiment_score'] = 0.0
            return articles
            
        async def storage_async(articles):
            # Імітація асинхронного збереження
            return articles
        
        # Встановлюємо side effects для моків
        mock_crawler.side_effect = crawler_async
        mock_scraper.side_effect = scraper_async
        mock_translator.side_effect = translator_async
        mock_summarizer.side_effect = summarizer_async
        mock_sentiment.side_effect = sentiment_async
        mock_storage.side_effect = storage_async
        
        # Мокуємо HTMLReporterAgent.arun
        with patch('main.HTMLReporterAgent') as mock_html:
            mock_html_instance = mock_html.return_value
            
            async def html_async(articles):
                return articles
                
            mock_html_instance.arun.side_effect = html_async
            
            # Запускаємо асинхронний режим
            result = asyncio.run(self.news_system.arun())
        
        # Перевірка, що всі асинхронні методи були викликані
        mock_crawler.assert_called_once()
        mock_translator.assert_called_once()
        mock_summarizer.assert_called_once()
        mock_sentiment.assert_called_once()
        mock_storage.assert_called_once()

class TestEndToEndFlow(unittest.TestCase):
    """Інтеграційні тести повного потоку обробки новин з реальними даними"""
    
    @unittest.skip("Цей тест залежить від реальних зовнішніх сервісів, тому вимкнений за замовчуванням")
    def test_real_integration(self):
        """Реальний інтеграційний тест з одним RSS-каналом та без API ключів"""
        # Створюємо тимчасові файли для тесту
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()
        
        temp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        temp_html.close()
        
        # Конфігурація для реального тесту
        config = {
            'rss_feeds': [
                'https://feeds.bbci.co.uk/news/technology/rss.xml'
            ],
            'categories': {
                'AI та машинне навчання': ['AI', 'штучний інтелект', 'machine learning', 'artificial intelligence'],
                'Data Science': ['data', 'аналітика', 'big data', 'analytics']
            },
            'scrape_urls': [],  # Без скрапінгу веб-сторінок
            'ai_queries': [],   # Без AI-запитів
            'db_path': temp_db.name,
            'html_report_file': temp_html.name
        }
        
        # Створюємо систему
        news_system = AINewsSystem(config)
        
        try:
            # Запускаємо систему
            news_system.run()
            
            # Перевіряємо, що HTML-звіт було створено
            self.assertTrue(os.path.exists(temp_html.name))
            
            # Перевіряємо, що статті збережено в базу даних
            conn = sqlite3.connect(temp_db.name)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            conn.close()
            
            # Повинні бути хоча б кілька статей
            self.assertGreater(count, 0)
            
        finally:
            # Видалення тимчасових файлів
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
            if os.path.exists(temp_html.name):
                os.unlink(temp_html.name)

if __name__ == '__main__':
    unittest.main() 
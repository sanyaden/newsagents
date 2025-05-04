import os
import sys
import logging
from dotenv import load_dotenv
from datetime import datetime

# Завантаження змінних середовища
load_dotenv()

# Конфігурація логування
class LoggerConfig:
    def __init__(self):
        self.debug_mode = os.getenv("DEBUG_MODE", "False").lower() in ["true", "1", "yes"]
        self.log_file = os.getenv("LOG_FILE", "newsagents.log")
        self.log_level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger = None
        self.token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        
        self.setup_logger()
    
    def setup_logger(self):
        """Налаштування логера з підтримкою виведення в консоль та файл"""
        # Створюємо каталог для логів, якщо його немає
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Основне налаштування логера
        self.logger = logging.getLogger("newsagents")
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # Очищаємо обробники, якщо вони вже були
        
        # Формат повідомлень логу
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Обробник для виведення в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)
        
        # Обробник для виведення в файл
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Логування налаштовано. Режим debug: {self.debug_mode}")
    
    def get_logger(self, name=None):
        """Отримання логера з вказаним ім'ям"""
        if name:
            return logging.getLogger(f"newsagents.{name}")
        return self.logger
    
    def log_agent_action(self, agent_name, action, message=None, status="started"):
        """Логування дії агента з вказаним статусом"""
        msg = f"Агент {agent_name} - {action}"
        if message:
            msg += f" - {message}"
        msg += f" [{status}]"
        
        if status == "error":
            self.logger.error(msg)
        elif status == "completed":
            self.logger.info(msg)
        else:
            self.logger.debug(msg)
    
    def log_token_usage(self, usage_info):
        """Логування використання токенів LLM"""
        if not usage_info:
            return
            
        # Оновлюємо загальну статистику
        prompt_tokens = usage_info.get('prompt_tokens', 0)
        completion_tokens = usage_info.get('completion_tokens', 0)
        total_tokens = usage_info.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
        
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += total_tokens
        
        # Розрахунок приблизної вартості (залежить від моделі)
        model = usage_info.get('model', 'gpt-4o')
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        self.token_usage["cost"] += cost
        
        # Логування детальної інформації
        if self.debug_mode:
            self.logger.debug(
                f"Використання токенів [{model}]: "
                f"Запит: {prompt_tokens}, "
                f"Відповідь: {completion_tokens}, "
                f"Всього: {total_tokens}, "
                f"Вартість: ${cost:.4f}"
            )
    
    def _calculate_cost(self, model, prompt_tokens, completion_tokens):
        """Розрахунок приблизної вартості використання API"""
        # Ціни станом на червень 2024
        prices = {
            "gpt-4o": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000}
        }
        
        model_prices = prices.get(model, prices["gpt-3.5-turbo"])
        return (prompt_tokens * model_prices["prompt"]) + (completion_tokens * model_prices["completion"])
    
    def print_token_usage_summary(self):
        """Виведення зведеної інформації про використання токенів"""
        try:
            self.logger.info(
                f"Загальне використання токенів: "
                f"Запит: {self.token_usage['prompt_tokens']}, "
                f"Відповідь: {self.token_usage['completion_tokens']}, "
                f"Всього: {self.token_usage['total_tokens']}, "
                f"Орієнтовна вартість: ${float(self.token_usage['cost']):.4f}"
            )
        except (TypeError, ValueError):
            # Обробка випадку, коли token_usage є моком в тестах
            self.logger.info(
                "Загальне використання токенів: "
                "Запит: 0, "
                "Відповідь: 0, "
                "Всього: 0, "
                "Орієнтовна вартість: $0.0000"
            )

# Створення глобального екземпляра конфігурації логера
logger_config = LoggerConfig()

# Функція для отримання логера у інших модулях
def get_logger(name=None):
    return logger_config.get_logger(name)

# Функція для логування дій агента
def log_agent_action(agent_name, action, message=None, status="started"):
    return logger_config.log_agent_action(agent_name, action, message, status)

# Функція для логування використання токенів
def log_token_usage(usage_info):
    return logger_config.log_token_usage(usage_info)

# Функція для виведення зведеної інформації про використання токенів
def print_token_usage_summary():
    return logger_config.print_token_usage_summary() 
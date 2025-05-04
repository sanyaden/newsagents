import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Додаємо шлях до батьківської директорії, щоб імпортувати модулі проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Імпортуємо модулі для тестування
from logger_config import get_logger, log_agent_action, log_token_usage, print_token_usage_summary

class TestLogger(unittest.TestCase):
    """Тести для системи логування"""
    
    def setUp(self):
        """Підготовка до тестів"""
        # Зберігаємо початкове значення DEBUG_MODE
        self.original_debug_mode = os.environ.get('DEBUG_MODE', None)
    
    def tearDown(self):
        """Очищення після тестів"""
        # Відновлюємо початкове значення DEBUG_MODE
        if self.original_debug_mode is not None:
            os.environ['DEBUG_MODE'] = self.original_debug_mode
        elif 'DEBUG_MODE' in os.environ:
            del os.environ['DEBUG_MODE']
    
    def test_logger_initialization(self):
        """Тестування ініціалізації логера"""
        logger = get_logger('test_logger')
        self.assertIsNotNone(logger)
        
        # Перевіряємо, що ім'я логера правильне
        self.assertEqual(logger.name, 'newsagents.test_logger')
    
    def test_debug_mode_setting(self):
        """Тестування налаштування режиму debug"""
        # Вже існує глобальний екземпляр у logger_config, тому ми перевіряємо ефективний рівень логування
        # без повторної ініціалізації
        
        # Отримуємо поточний логер
        logger = get_logger('test')
        current_level = logger.getEffectiveLevel()
        
        # Перевіряємо, що рівень логування відповідає режиму DEBUG_MODE з .env
        debug_mode_env = os.environ.get('DEBUG_MODE', 'False').lower() in ["true", "1", "yes"]
        expected_level = 10 if debug_mode_env else 20  # 10 = DEBUG, 20 = INFO
        
        self.assertEqual(current_level, expected_level)
    
    @patch('logger_config.logger_config.logger')
    def test_log_agent_action(self, mock_logger):
        """Тестування логування дій агента"""
        # Тестування логування з різними статусами
        log_agent_action('TestAgent', 'test_action', 'тестове повідомлення', 'started')
        mock_logger.debug.assert_called_once()
        
        mock_logger.reset_mock()
        log_agent_action('TestAgent', 'test_action', 'тестове повідомлення', 'completed')
        mock_logger.info.assert_called_once()
        
        mock_logger.reset_mock()
        log_agent_action('TestAgent', 'test_action', 'тестове повідомлення', 'error')
        mock_logger.error.assert_called_once()
    
    @patch('logger_config.logger_config.debug_mode', True)
    @patch('logger_config.logger_config.logger')
    def test_log_token_usage(self, mock_logger):
        """Тестування логування використання токенів"""
        # Тестові дані про використання токенів
        token_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'model': 'gpt-3.5-turbo'
        }
        
        # Логування використання токенів
        log_token_usage(token_usage)
        mock_logger.debug.assert_called_once()
        
        # Перевіряємо повторне логування з іншою моделлю
        mock_logger.reset_mock()
        token_usage['model'] = 'gpt-4o'
        token_usage['prompt_tokens'] = 200
        token_usage['completion_tokens'] = 100
        token_usage['total_tokens'] = 300
        
        log_token_usage(token_usage)
        mock_logger.debug.assert_called_once()
    
    @patch('logger_config.logger_config.logger')
    @patch('logger_config.logger_config.token_usage')
    def test_print_token_usage_summary(self, mock_token_usage, mock_logger):
        """Тестування виведення зведеної інформації про використання токенів"""
        # Встановлюємо вартість як числове значення, а не MagicMock
        mock_token_usage.__getitem__.side_effect = lambda key: {
            'prompt_tokens': 300,
            'completion_tokens': 150,
            'total_tokens': 450,
            'cost': 0.0036
        }[key]
        
        # Виведення зведеної інформації
        print_token_usage_summary()
        mock_logger.info.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
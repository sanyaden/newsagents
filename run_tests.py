#!/usr/bin/env python3
"""
Скрипт для запуску модульних тестів NewsAgents
Використовує окремий файл конфігурації .env.test
"""

import os
import sys
import unittest
import shutil
import tempfile

# Встановлюємо змінні середовища для тестів
def setup_test_env():
    """Налаштування середовища для тестування"""
    # Читаємо .env.test файл
    env_vars = {}
    try:
        with open('.env.test', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value
                    os.environ[key] = value
        
        print(f"Завантажено налаштування тестування з .env.test")
    except Exception as e:
        print(f"Помилка завантаження .env.test: {e}")
    
    # Встановлюємо додаткові змінні середовища
    os.environ['TESTING'] = 'True'
    
    # Створюємо тимчасову базу даних
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    os.environ['DB_PATH'] = temp_db.name
    
    # Створюємо тимчасовий HTML-звіт
    temp_html = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
    temp_html.close()
    os.environ['OUTPUT_FILE'] = temp_html.name
    
    return {
        'temp_db': temp_db.name,
        'temp_html': temp_html.name
    }

def cleanup_test_env(temp_files):
    """Очищення тимчасових файлів після тестування"""
    for file_path in temp_files.values():
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                print(f"Видалено тимчасовий файл: {file_path}")
            except Exception as e:
                print(f"Помилка видалення файлу {file_path}: {e}")

def run_tests():
    """Запуск всіх тестів"""
    # Завантажуємо всі тести з директорії tests
    test_suite = unittest.defaultTestLoader.discover('tests')
    
    # Запускаємо тести
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Повертаємо код виходу
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Налаштування середовища
    temp_files = setup_test_env()
    
    try:
        # Запуск тестів
        exit_code = run_tests()
        
        # Вихід з відповідним кодом
        sys.exit(exit_code)
    finally:
        # Очищення середовища
        cleanup_test_env(temp_files) 
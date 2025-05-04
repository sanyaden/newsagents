"""
Список українських джерел для скрапінгу новин.
Використовується в системі NewsAgents для збору новин з українських технологічних медіа.
"""

# IT та технологічні ресурси
IT_TECH_SOURCES = [
    {
        "url": "https://dou.ua",
        "name": "DOU",
        "category": "IT та технології",
        "description": "Спільнота розробників України",
        "article_selector": "article.b-typo, .b-similar-articles .item, .b-articles-list article"
    },
    {
        "url": "https://dev.ua",
        "name": "Dev.ua",
        "category": "IT та технології",
        "description": "Українське IT-медіа про бізнес та технології",
        "article_selector": ".feed-item, .new-card-design"
    },
    {
        "url": "https://ain.ua",
        "name": "AIN.UA",
        "category": "IT та технології",
        "description": "Найбільше інтернет-видання про IT-бізнес в Україні",
        "article_selector": ".post-item"
    },
    {
        "url": "https://mc.today",
        "name": "MC Today",
        "category": "IT та технології",
        "description": "Видання про підприємництво, технології та інновації",
        "article_selector": ".mc-card, .post, .post-item"
    },
    {
        "url": "https://speka.media",
        "name": "Speka Media",
        "category": "IT та технології",
        "description": "Медіа про технологічне підприємництво",
        "article_selector": ".article-preview"
    },
    {
        "url": "https://highload.tech",
        "name": "Highload",
        "category": "IT та технології",
        "description": "Спеціалізоване видання про архітектуру розробки та операційну діяльність",
        "article_selector": ".archive-post"
    },
    {
        "url": "https://itc.ua",
        "name": "ITC.ua",
        "category": "IT та технології",
        "description": "Новини технологій, огляди гаджетів, IT-індустрія",
        "article_selector": ".entry-item"
    }
]

# Загальні новини/ЗМІ з технологічними розділами
GENERAL_MEDIA_TECH_SECTIONS = [
    {
        "url": "https://unian.ua/science",
        "name": "УНІАН Наука і технології",
        "category": "Технології та наука",
        "description": "Розділ науки і технологій інформаційного агентства УНІАН",
        "article_selector": ".list-thumbs__item"
    },
    {
        "url": "https://24tv.ua/tech",
        "name": "24 Канал Техно",
        "category": "Технології та наука",
        "description": "Розділ технологій 24 Каналу",
        "article_selector": ".news-list__item"
    },
    {
        "url": "https://espreso.tv/tekhnologiyi",
        "name": "Еспресо TV Технології",
        "category": "Технології та наука",
        "description": "Розділ технологій телеканалу Еспресо",
        "article_selector": ".article-preview"
    },
    {
        "url": "https://focus.ua/uk/technologies",
        "name": "Focus Технології",
        "category": "Технології та наука",
        "description": "Розділ технологій журналу Focus",
        "article_selector": ".c-card"
    },
    {
        "url": "https://www.ukr.net/news/technologies.html",
        "name": "Ukr.net Технології",
        "category": "Технології та наука",
        "description": "Агрегатор новин технологій на Ukr.net",
        "article_selector": ".im-tl"
    },
    {
        "url": "https://nauka.ua",
        "name": "Наука.UA",
        "category": "Технології та наука",
        "description": "Науково-популярне видання",
        "article_selector": ".article-preview, .post"
    },
    {
        "url": "https://shotam.info/category/texnologiyi/",
        "name": "ШоТам Технології",
        "category": "Технології та наука",
        "description": "Розділ технологій медіа ШоТам",
        "article_selector": ".post, .post-item"
    }
]

# Повний список всіх джерел
ALL_UKRAINIAN_SOURCES = IT_TECH_SOURCES + GENERAL_MEDIA_TECH_SECTIONS

# Список тільки URL-адрес для скрапінгу (для використання у конфігурації)
UKRAINIAN_SCRAPING_URLS = [source["url"] for source in ALL_UKRAINIAN_SOURCES]

if __name__ == "__main__":
    print("Українські джерела для скрапінгу:")
    
    print("\nIT та технологічні ресурси:")
    for idx, source in enumerate(IT_TECH_SOURCES, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print("\nЗагальні ЗМІ з технологічними розділами:")
    for idx, source in enumerate(GENERAL_MEDIA_TECH_SECTIONS, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print(f"\nЗагальна кількість джерел: {len(ALL_UKRAINIAN_SOURCES)}") 
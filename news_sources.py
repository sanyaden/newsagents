"""
Список джерел для скрапінгу новин.
Використовується в системі NewsAgents для збору новин з українських 
та міжнародних технологічних медіа.
"""

# IT та технологічні ресурси України
IT_TECH_SOURCES_UA = [
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

# Загальні новини/ЗМІ України з технологічними розділами
GENERAL_MEDIA_TECH_SECTIONS_UA = [
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

# Міжнародні технологічні медіа
INTERNATIONAL_TECH_SOURCES = [
    {
        "url": "https://techcrunch.com",
        "name": "TechCrunch",
        "category": "Технології та інновації",
        "description": "Медіа про технології та стартапи",
        "article_selector": "article, .post-block"
    },
    {
        "url": "https://www.wired.com",
        "name": "Wired",
        "category": "Технології та наука",
        "description": "Актуальні новини та аналітика про технології",
        "article_selector": ".summary-item, article"
    },
    {
        "url": "https://www.theverge.com",
        "name": "The Verge",
        "category": "Технології та наука",
        "description": "Новини про гаджети та технології",
        "article_selector": "article, .c-entry-box--compact"
    },
    {
        "url": "https://arstechnica.com",
        "name": "Ars Technica",
        "category": "Технології та наука",
        "description": "Технології, ІТ, наука",
        "article_selector": "article, .article"
    },
    {
        "url": "https://www.cnet.com",
        "name": "CNET",
        "category": "Технології та гаджети",
        "description": "Огляди продуктів та новини технологій",
        "article_selector": "article, .c-shortcodeListicle"
    },
    {
        "url": "https://gizmodo.com",
        "name": "Gizmodo",
        "category": "Технології та наука",
        "description": "Дизайн, технології, наука і науково-популярна фантастика",
        "article_selector": "article, .entry-header"
    },
    {
        "url": "https://www.zdnet.com",
        "name": "ZDNet",
        "category": "Технології та бізнес",
        "description": "Бізнес-технології, ІТ, трансформація",
        "article_selector": "article, .item, .river-post"
    },
    {
        "url": "https://www.techradar.com",
        "name": "TechRadar",
        "category": "Технології та гаджети",
        "description": "Огляди продуктів та новини технологій",
        "article_selector": "article, .listingResult"
    },
    {
        "url": "https://www.engadget.com",
        "name": "Engadget",
        "category": "Технології та гаджети",
        "description": "Технології споживчої електроніки",
        "article_selector": "article, .o-hit"
    },
    {
        "url": "https://www.reuters.com/technology",
        "name": "Reuters Technology",
        "category": "Технології та бізнес",
        "description": "Технологічні новини від міжнародного інформагентства",
        "article_selector": "article, .story"
    },
    {
        "url": "https://www.geekwire.com",
        "name": "GeekWire",
        "category": "Технології та стартапи",
        "description": "Технологічні новини, стартапи, ІТ",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.computerworld.com",
        "name": "ComputerWorld",
        "category": "ІТ та технології",
        "description": "Корпоративні ІТ, технологічний бізнес",
        "article_selector": "article, .item"
    },
    {
        "url": "https://www.theregister.com",
        "name": "The Register",
        "category": "ІТ та технології",
        "description": "Новини ІТ-індустрії",
        "article_selector": "article, .story"
    },
    {
        "url": "https://www.digitaltrends.com",
        "name": "Digital Trends",
        "category": "Технології та гаджети",
        "description": "Огляди пристроїв та технологічні тренди",
        "article_selector": "article, .b-small-article"
    },
    {
        "url": "https://thenextweb.com",
        "name": "The Next Web",
        "category": "Технології та інновації",
        "description": "Міжнародні технологічні новини та бізнес",
        "article_selector": "article, .story"
    }
]

# Джерела новин про дані та аналітику
DATA_SCIENCE_SOURCES = [
    {
        "url": "https://www.analyticsinsight.net",
        "name": "Analytics Insight",
        "category": "Аналітика даних",
        "description": "Штучний інтелект та аналітика даних",
        "article_selector": "article, .jeg_post"
    },
    {
        "url": "https://www.kdnuggets.com",
        "name": "KDnuggets",
        "category": "Data Science",
        "description": "Аналітика даних, Data Science, машинне навчання",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.technologyreview.com",
        "name": "MIT Technology Review",
        "category": "Технології та інновації",
        "description": "Технологічні інновації та їх вплив",
        "article_selector": "article, .feedItem"
    },
    {
        "url": "https://insidebigdata.com",
        "name": "insideBIGDATA",
        "category": "Big Data",
        "description": "Новини та аналітика Big Data",
        "article_selector": "article, .item"
    },
    {
        "url": "https://www.dataversity.net",
        "name": "DATAVERSITY",
        "category": "Дані та аналітика",
        "description": "Керування даними, Big Data, аналітика",
        "article_selector": "article, .post"
    },
    {
        "url": "https://datafloq.com",
        "name": "Datafloq",
        "category": "Дані та аналітика",
        "description": "Big Data, блокчейн, AI",
        "article_selector": "article, .post"
    },
    {
        "url": "https://dataconomy.com",
        "name": "Dataconomy",
        "category": "Дані та аналітика",
        "description": "Дані, технології, інновації",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.datanami.com",
        "name": "Datanami",
        "category": "Дані та аналітика",
        "description": "Big Data, аналітика даних, інфраструктура",
        "article_selector": "article, .story"
    },
    {
        "url": "https://www.datasciencecentral.com",
        "name": "Data Science Central",
        "category": "Data Science",
        "description": "Спільнота з Data Science, машинного навчання, ІІ",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.dbta.com",
        "name": "Database Trends and Applications",
        "category": "Бази даних та технології",
        "description": "Бази даних, аналітика даних, Big Data",
        "article_selector": "article, .article-item"
    },
    {
        "url": "https://www.informs.org",
        "name": "INFORMS",
        "category": "Аналітика та дослідження операцій",
        "description": "Аналітика, дослідження операцій, наука про дані",
        "article_selector": "article, .news-item"
    },
    {
        "url": "https://www.analyticsvidhya.com",
        "name": "Analytics Vidhya",
        "category": "Data Science",
        "description": "Аналітика даних, машинне навчання, Python",
        "article_selector": "article, .post"
    }
]

# Джерела новин про штучний інтелект
AI_SOURCES = [
    {
        "url": "https://distill.pub",
        "name": "Distill",
        "category": "Штучний інтелект",
        "description": "Машинне навчання, дослідження",
        "article_selector": "article, .post"
    },
    {
        "url": "https://developer.ibm.com",
        "name": "IBM Developer",
        "category": "Розробка та ШІ",
        "description": "AI, хмарні технології, розробка",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.aitrends.com",
        "name": "AI Trends",
        "category": "Штучний інтелект",
        "description": "Тренди та новини у сфері AI",
        "article_selector": "article, .post"
    },
    {
        "url": "https://intelligence.org",
        "name": "Machine Intelligence Research Institute",
        "category": "Штучний інтелект",
        "description": "Дослідження AI безпеки",
        "article_selector": "article, .post"
    },
    {
        "url": "https://a16z.com/ai-playbook",
        "name": "A16Z AI Playbook",
        "category": "Штучний інтелект",
        "description": "Практичні вказівки щодо застосування ШІ",
        "article_selector": "article, .post"
    },
    {
        "url": "https://news.mit.edu",
        "name": "MIT News",
        "category": "Технології та наука",
        "description": "Новини з світу науки та технологій від MIT",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.sciencedaily.com",
        "name": "ScienceDaily",
        "category": "Наука та технології",
        "description": "Новини науки, включаючи AI та технології",
        "article_selector": "article, .latest-head"
    },
    {
        "url": "https://emerj.com",
        "name": "Emerj",
        "category": "Штучний інтелект",
        "description": "Дослідження AI та його впливу на бізнес",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.r-bloggers.com",
        "name": "R-Bloggers",
        "category": "R та аналітика даних",
        "description": "Статті та туторіали про R",
        "article_selector": "article, .post"
    },
    {
        "url": "https://jair.org",
        "name": "Journal of Artificial Intelligence Research",
        "category": "Штучний інтелект",
        "description": "Академічні дослідження в області AI",
        "article_selector": "article, .post"
    },
    {
        "url": "https://jaxenter.com",
        "name": "JAXenter",
        "category": "Java та технології",
        "description": "Java, ІТ, ШІ та веб-розробка",
        "article_selector": "article, .post-small"
    },
    {
        "url": "https://hunch.net",
        "name": "Hunch.net",
        "category": "Машинне навчання",
        "description": "Блог про машинне навчання",
        "article_selector": "article, .post"
    },
    {
        "url": "https://www.artificialintelligence-news.com",
        "name": "AI News",
        "category": "Штучний інтелект",
        "description": "Новини ШІ та машинного навчання",
        "article_selector": "article, .post"
    },
    {
        "url": "https://aimagazine.com",
        "name": "AI Magazine",
        "category": "Штучний інтелект",
        "description": "Журнал про ШІ та його застосування",
        "article_selector": "article, .post"
    }
]

# Повний список всіх джерел
ALL_UKRAINIAN_SOURCES = IT_TECH_SOURCES_UA + GENERAL_MEDIA_TECH_SECTIONS_UA
ALL_INTERNATIONAL_SOURCES = INTERNATIONAL_TECH_SOURCES + DATA_SCIENCE_SOURCES + AI_SOURCES
ALL_SOURCES = ALL_UKRAINIAN_SOURCES + ALL_INTERNATIONAL_SOURCES

# Список тільки URL-адрес для скрапінгу (для використання у конфігурації)
UKRAINIAN_SCRAPING_URLS = [source["url"] for source in ALL_UKRAINIAN_SOURCES]
INTERNATIONAL_SCRAPING_URLS = [source["url"] for source in ALL_INTERNATIONAL_SOURCES]
ALL_SCRAPING_URLS = UKRAINIAN_SCRAPING_URLS + INTERNATIONAL_SCRAPING_URLS

if __name__ == "__main__":
    print("Джерела для скрапінгу новин:")
    
    print("\nУкраїнські IT та технологічні ресурси:")
    for idx, source in enumerate(IT_TECH_SOURCES_UA, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print("\nУкраїнські ЗМІ з технологічними розділами:")
    for idx, source in enumerate(GENERAL_MEDIA_TECH_SECTIONS_UA, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print("\nМіжнародні технологічні медіа:")
    for idx, source in enumerate(INTERNATIONAL_TECH_SOURCES, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print("\nДжерела новин про дані та аналітику:")
    for idx, source in enumerate(DATA_SCIENCE_SOURCES, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print("\nДжерела новин про штучний інтелект:")
    for idx, source in enumerate(AI_SOURCES, 1):
        print(f"{idx}. {source['name']} ({source['url']})")
    
    print(f"\nЗагальна кількість джерел: {len(ALL_SOURCES)}")
    print(f"Українські джерела: {len(ALL_UKRAINIAN_SOURCES)}")
    print(f"Міжнародні джерела: {len(ALL_INTERNATIONAL_SOURCES)}") 
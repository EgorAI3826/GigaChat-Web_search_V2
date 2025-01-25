import argparse
import json
import logging
import requests
import sys
import threading
import time
from datetime import datetime
from duckduckgo_search import DDGS
from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse
from typing import Dict, List, Union

APPNAME = 'GigaSearch_v3'
VERSION = '0.7'

BINDING_ADDRESS = '127.0.0.1'
BINDING_PORT = 5000
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'infidelis/GigaChat-20B-A3B-instruct:bf16'
API_TO_USE = 'ollama'
SILENT = False
SEARCH_RESULT_COUNT = 7
NEWS_RESULT_COUNT = 5
TRIM_WIKIPEDIA_SUMMARY = True
TRIM_WIKIPEDIA_LINES = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

source_links = []
SEARCH_TYPE = ""

class SearchParser:
    @staticmethod
    def parse_web_result(result: Dict) -> Dict:
        parsed = urlparse(result['href'])
        return {
            'title': result['title'],
            'url': result['href'],
            'domain': parsed.netloc,
            'path': parsed.path,
            'snippet': result['body'],
            'favicon': f"https://{parsed.netloc}/favicon.ico",
            'keywords': result.get('keywords', []),
            'language': result.get('language', 'en'),
            'meta_description': result.get('meta_description', ''),
            'meta_keywords': result.get('meta_keywords', ''),
            'content_length': result.get('content_length', 0),
            'content_type': result.get('content_type', ''),
            'last_modified': result.get('last_modified', ''),
            'server': result.get('server', ''),
            'encoding': result.get('encoding', ''),
            'links': result.get('links', [])
        }

    @staticmethod
    def parse_news_result(result: Dict) -> Dict:
        return {
            'title': result['title'],
            'url': result['url'],
            'source': result['source'],
            'date': result.get('date', 'N/A'),
            'snippet': result['body'],
            'thumbnail': result.get('thumbnail', ''),
            'author': result.get('author', ''),
            'category': result.get('category', ''),
            'tags': result.get('tags', []),
            'content': result.get('content', ''),
            'language': result.get('language', 'en'),
            'location': result.get('location', ''),
            'related_articles': result.get('related_articles', [])
        }

def search(search_query: str, num_results: int) -> List[Dict]:
    try:
        results = DDGS().text(search_query, max_results=num_results+3)
        processed = []
        for result in results[:num_results]:
            parsed = SearchParser.parse_web_result(result)
            processed.append(parsed)
            source_links.append(parsed['url'])
        logger.debug(f"Поисковые результаты: {processed}")
        return processed
    except Exception as e:
        logger.error(f"Ошибка поиска: {str(e)}")
        return []

def news(search_query: str, num_results: int) -> List[Dict]:
    try:
        results = DDGS().news(search_query, max_results=num_results+3)
        processed = []
        for result in results[:num_results]:
            parsed = SearchParser.parse_news_result(result)
            processed.append(parsed)
            source_links.append(parsed['url'])
        logger.debug(f"Новостные результаты: {processed}")
        return processed
    except Exception as e:
        logger.error(f"Ошибка новостей: {str(e)}")
        return []

class WikipediaClient:
    @staticmethod
    def get_summary(page_title: str) -> Dict:
        url = "https://ru.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts|info',
            'exintro': True,
            'explaintext': True,
            'inprop': 'url',
            'titles': page_title
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            page = next(iter(data['query']['pages'].values()))
            if 'extract' not in page:
                return {'title': page['title'], 'summary': 'Нет данных', 'url': f"https://ru.wikipedia.org/?curid={page['pageid']}"}
            return {
                'title': page['title'],
                'summary': '.'.join(page['extract'].split('.')[:TRIM_WIKIPEDIA_LINES]) + '.' if TRIM_WIKIPEDIA_SUMMARY else page['extract'],
                'url': f"https://ru.wikipedia.org/?curid={page['pageid']}",
                'categories': page.get('categories', []),
                'pageviews': page.get('pageviews', 0),
                'revisions': page.get('revisions', 0),
                'links': page.get('links', []),
                'references': page.get('references', [])
            }
        except Exception as e:
            logger.error(f"Ошибка Википедии: {str(e)}")
            return {'title': 'N/A', 'summary': 'Нет данных', 'url': ''}

def wikipedia(search_arg: str) -> Dict:
    try:
        results = DDGS().text(f"site:wikipedia.org {search_arg}", max_results=1)
        if not results:
            return {}
        wiki_url = results[0]['href']
        source_links.append(wiki_url)
        page_title = urlparse(wiki_url).path.split('/')[-1]
        return WikipediaClient.get_summary(page_title)
    except Exception as e:
        logger.error(f"Ошибка поиска Википедии: {str(e)}")
        return {}

def generate_search_query(user_query: str) -> str:
    prompt = (
        "Ты — помощник, который формирует поисковые запросы для Google. "
        "На основе сообщения пользователя сформулируй подробный и точный запрос для поиска в интернете. "
        "Учитывай ключевые слова, контекст и суть запроса. "
        "Вот сообщение пользователя:\n\n"
        f"{user_query}\n\n"
        "Сформируй подробный поисковый запрос для Google: ОН ДОЛЖЕН БЫТЬ КРАТКИМ!"
    )
    response = ollama_query(prompt)
    if response and 'response' in response:
        return response['response'].strip()
    return user_query

def perform_searches(query: str) -> Dict:
    logger.info("Сбор данных...")
    return {
        'wikipedia': wikipedia(query),
        'web': search(query, SEARCH_RESULT_COUNT),
        'news': news(query, NEWS_RESULT_COUNT)
    }

def format_results(data: Dict) -> str:
    output = []

    if data['wikipedia'].get('title'):
        output.append(
            f"Википедия: {data['wikipedia']['title']}\n"
            f"{data['wikipedia']['summary']}\n"
            f"Источник: {data['wikipedia']['url']}\n"
        )

    if data['web']:
        output.append("Веб-результаты:")
        for idx, result in enumerate(data['web'], 1):
            output.append(
                f"{idx}. [{result['domain']}] {result['title']}\n"
                f"   {result['snippet']}\n"
                f"   {result['url']}\n"
            )

    if data['news']:
        output.append("Новости:")
        for idx, news_item in enumerate(data['news'], 1):
            output.append(
                f"{idx}. {news_item['source']} ({news_item['date']})\n"
                f"   {news_item['title']}\n"
                f"   {news_item['snippet']}\n"
                f"   {news_item['url']}\n"
            )

    return '\n'.join(output)

def ollama_query(prompt: str) -> Union[Dict, None]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False}
        )
        return response.json()
    except Exception as e:
        logger.error(f"Ошибка API: {str(e)}")
        return None

def _is_llama_online() -> bool:
    try:
        response = requests.get(OLLAMA_BASE_URL)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ошибка подключения к Ollama: {e}")
        return False

def process_query(search_query: str) -> str:
    if not _is_llama_online():
        return "Ошибка: Сервер ИИ недоступен."

    start_time = time.time()
    search_query = generate_search_query(search_query)
    logger.info(f"Сформированный поисковый запрос: {search_query}")

    data = perform_searches(search_query)
    formatted_data = format_results(data)
    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = (
        "Предоставь подробный ответ на запрос, основываясь на следующих источниках. "
        "Будь точным, информативным и полезным. "
        "Ссылайся на источники как [1], [2] или [3] после каждого предложения (не только в конце), "
        "чтобы подтвердить ответ (Пример: Правильно: [1], Правильно: [2][3], Неправильно: [1, 2]).\n\n"
        f"Запрос: {search_query}\n\n"
        f"Источники:\n{formatted_data}\n\n"
        f"Текущая дата: {current_date_time}\n\n"
        "Ответ:"
    )

    logger.info("Анализ данных...")
    response = ollama_query(prompt)

    if not response or 'response' not in response:
        return "Ошибка обработки запроса."

    processing_time = time.time() - start_time
    sources = list(set(source_links))
    sources_text = "\n".join(f"{idx + 1}. {link}" for idx, link in enumerate(sources))
    return (
        f"Запрос: {search_query}\n\n"
        f"Результаты анализа ({processing_time:.2f} сек):\n\n"
        f"{response['response']}\n\n"
        f"Источники:\n{sources_text}"
    )

def web_server():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/search', methods=['POST'])
    def handle_search():
        if lock.locked():
            return jsonify({'error': 'Система занята'}), 503

        with lock:
            query = request.form.get('query', '')
            if not query:
                return jsonify({'error': 'Пустой запрос'}), 400

            result = process_query(query)
            return jsonify({
                'query': query,
                'result': result,
                'sources': list(set(source_links))
            })

    logger.info(f"Сервер запущен на {BINDING_ADDRESS}:{BINDING_PORT}")
    app.run(host=BINDING_ADDRESS, port=BINDING_PORT)

def cli_interface(query: str):
    print(f"\nОбработка запроса: {query}")
    result = process_query(query)
    print(f"\nРезультат:\n{result}")

def main():
    parser = argparse.ArgumentParser(description=f"{APPNAME} v{VERSION}")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-q', '--query', help="Поисковый запрос")
    group.add_argument('-s', '--server', action='store_true', help="Запустить веб-сервер")

    args = parser.parse_args()

    if args.server:
        SEARCH_TYPE = "web"
        web_server()
    else:
        SEARCH_TYPE = "cli"
        if not args.query:
            print("Ошибка: Необходимо указать запрос с помощью -q/--query")
            sys.exit(1)
        cli_interface(args.query)

if __name__ == "__main__":
    lock = threading.Lock()
    main()
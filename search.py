from duckduckgo_search import DDGS
from seleniumbase import Driver
from bs4 import BeautifulSoup
import ollama
import logging
import concurrent.futures
import time
import random

# Made by EgorAI3826
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "query_optimize": "Сгенерируй точный поисковый запрос для: '{query}'. Только запрос.",
    "content_summary": "Выдели информацию связанную с '{query}' из текста. Текст: {text}",
    "final_answer": (
        "Предоставь ответ на запрос '{query}' длиной 5-8 предложений, используя приведенные данные. "
        "Будь оригинальным, точным и полезным. В конце ответа добавь раздел 'Источники:' с нумерованным списком ссылок. "
        "Текущее время: {current_time}. Данные: {context}"
    )
}

MODEL_NAME = "infidelis/GigaChat-20B-A3B-instruct:bf16"

def log_execution(func):
    def wrapper(*args, **kwargs):
        logger.info(f"[START] {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"[END] {func.__name__} ({(time.time()-start):.2f}s)")
        return result
    return wrapper

@log_execution
def ai_query_optimizer(query):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': SYSTEM_PROMPTS["query_optimize"].format(query=query)
            }]
        )
        optimized = response['message']['content'].strip('"')
        logger.info(f"Оптимизированный запрос: {optimized}")
        return optimized
    except Exception as e:
        logger.error(f"Ошибка оптимизации: {str(e)}")
        return query

@log_execution
def ddg_links_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = [{
                'title': r['title'],
                'url': r['href'],
                'snippet': r['body']
            } for r in ddgs.text(query, max_results=max_results)]
            logger.info(f"Найдено результатов: {len(results)}")
            return results
    except Exception as e:
        logger.error(f"Ошибка поиска: {str(e)}")
        return []

def advanced_parser(url):
    driver = None
    try:
        driver = Driver(uc=True, headless=True)
        driver.get(url)
        time.sleep(random.uniform(0.5, 1.0))

        driver.execute_script("window.scrollBy(0, 500)")
        time.sleep(0.2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        content = ' '.join([
            el.get_text(separator=' ', strip=True)
            for el in soup.find_all(['p', 'h1', 'h2', 'h3'])
        ])
        logger.info(f"Ссылка: {url}\nДанные: {content[:100]}...")  # Логируем первые 100 символов
        return content  # Убрали ограничение длины
    except Exception as e:
        logger.error(f"Ошибка парсинга {url}: {str(e)}")
        return ""
    finally:
        if driver:
            driver.quit()

@log_execution
def parallel_parser(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(advanced_parser, url): url for url in urls}
        return [future.result() for future in concurrent.futures.as_completed(futures)]

@log_execution
def ai_content_processor(query, text):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': SYSTEM_PROMPTS["content_summary"].format(
                    query=query,
                    text=text  # Убрали обрезку текста
                )
            }]
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}")
        return ""

@log_execution
def build_final_response(query, sources):
    try:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        context = "\n".join([f"[{i+1}] {s['summary']}" for i, s in enumerate(sources)])
        urls = "\n".join([f"[{i+1}] {s['url']}" for i, s in enumerate(sources)])

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': SYSTEM_PROMPTS["final_answer"].format(
                    query=query,
                    current_time=current_time,
                    context=context + "\n\nСсылки на источники:\n" + urls
                )
            }]
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        return "Ошибка формирования ответа"

@log_execution
def main_pipeline(user_query):
    logger.info(f"\n{'='*50}\nСтарт обработки: {user_query}\n{'='*50}")

    optimized_query = ai_query_optimizer(user_query)
    search_results = ddg_links_search(optimized_query)

    urls = [result['url'] for result in search_results]
    parsed_contents = parallel_parser(urls)

    processed_sources = []
    for result, content in zip(search_results, parsed_contents):
        if content:
            summary = ai_content_processor(user_query, content)
            processed_sources.append({
                'url': result['url'],
                'summary': summary
            })

    final_response = build_final_response(user_query, processed_sources)
    logger.info(f"\n{'='*50}\nРезультат:\n{final_response}\n{'='*50}")
    return final_response

if __name__ == "__main__":
    user_question = "погода завтра в москве"
    print(main_pipeline(user_question))

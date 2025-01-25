
# GigaSearch_v2

GigaSearch_v2 — это мощный поисковый инструмент, который объединяет веб-поиск, новости и данные из Википедии. Проект использует API DuckDuckGo для поиска и модель Ollama для обработки запросов и формирования ответов. Поддерживает как командный интерфейс (CLI), так и веб-сервер.

## Основные возможности

- **Веб-поиск**: Поиск информации в интернете с использованием DuckDuckGo.
- **Новости**: Получение актуальных новостей по запросу.
- **Википедия**: Краткое изложение информации из Википедии.
- **Интеграция с Ollama**: Использование модели Ollama для формирования поисковых запросов и анализа результатов.
- **Веб-сервер**: Возможность запуска веб-сервера для обработки запросов через браузер.
- **Командный интерфейс**: Возможность использования через командную строку.

## Установка

1. Убедитесь, что у вас установлен Python 3.10 или выше.
2. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ваш-репозиторий/GigaSearch_v2.git
   cd GigaSearch_v2
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Зависимости

Основные зависимости:
- `flask` — для веб-сервера.
- `requests` — для HTTP-запросов.
- `duckduckgo-search` — для поиска через DuckDuckGo.
- `ollama` — для работы с моделью Ollama.

## Использование

### Запуск веб-сервера

Для запуска веб-сервера выполните:
```bash
python search.py --server
```
Сервер будет доступен по адресу: `http://127.0.0.1:5000`.

### Использование командной строки

Для поиска через командную строку выполните:
```bash
python search.py --query "ваш запрос"
```

### Пример запроса через веб-интерфейс

1. Откройте браузер и перейдите по адресу `http://127.0.0.1:5000`.
2. Введите запрос в поле поиска и нажмите "Поиск".
3. Результаты будут отображены на странице.

### Пример запроса через CLI

```bash
python search.py --query "Что такое искусственный интеллект?"
```

## Настройка

### Конфигурация

Основные параметры можно настроить в коде:
- `BINDING_ADDRESS` и `BINDING_PORT` — адрес и порт для веб-сервера.
- `OLLAMA_BASE_URL` и `OLLAMA_MODEL` — URL и модель Ollama.
- `SEARCH_RESULT_COUNT` и `NEWS_RESULT_COUNT` — количество результатов для веб-поиска и новостей.

### Логирование

Логи сохраняются в файл `logs.txt` и выводятся в консоль.

## Примеры использования

### Веб-поиск

```bash
python search.py --query "лучшие языки программирования 2024"
```

### Новости

```bash
python search.py --query "последние новости о технологиях"
```

### Википедия

```bash
python search.py --query "история Python"
```

## Лицензия

Этот проект распространяется под лицензией MIT. Подробнее см. в файле `LICENSE`.

## Автор

[egorai3826]

---

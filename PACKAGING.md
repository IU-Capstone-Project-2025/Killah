# 🚀 ИНСТРУКЦИЯ ПО УПАКОВКЕ ПРИЛОЖЕНИЯ

## Быстрый старт

1. **Соберите проект в Xcode** (⌘+B)
2. **Запустите упаковку**:
   ```bash
   cd "Killah Prototype"
   ./Scripts/run_packaging.sh
   ```

## Что происходит при упаковке

Скрипт `package_app.sh` выполняет следующие шаги:

1. 🔍 **Находит собранное приложение** в DerivedData
2. 🐍 **Копирует Python.framework** из системы в приложение
3. 📦 **Создает виртуальное окружение** внутри приложения
4. 📚 **Устанавливает зависимости** из requirements.txt
5. 📄 **Копирует Python скрипты** и модели
6. 🔧 **Патчит пути** к библиотекам
7. ✍️ **Переподписывает** исполняемые файлы

## Требования

- **Python 3.12** установленный с python.org (НЕ Homebrew!)
- **Xcode** с командными инструментами
- **Права администратора** для доступа к /Library/Frameworks

## Отладка

Скрипт выводит **МАКСИМАЛЬНО ПОДРОБНЫЕ** логи. Если что-то не работает:

1. Прочитайте весь вывод скрипта
2. Найдите строки с ❌ (ошибки)
3. Проверьте требования выше

## Ручной запуск (если автоматический не работает)

Если `run_packaging.sh` не находит сборку автоматически:

```bash
# 1. Найдите путь к сборке вручную
find ~/Library/Developer/Xcode/DerivedData -name "Killah Prototype.app" -type d

# 2. Экспортируйте путь к папке Build/Products/Debug
export BUILT_PRODUCTS_DIR="/path/to/Build/Products/Debug"

# 3. Запустите упаковку
./Scripts/package_app.sh
```

## Проверка результата

После успешной упаковки в приложении должны быть:
- `Contents/Frameworks/Python.framework/`
- `Contents/Resources/venv/`
- `Contents/Resources/autocomplete.py`
- `Contents/Resources/minillm_export.pt`

## Структура после упаковки

```
YourApp.app/
├── Contents/
│   ├── Frameworks/
│   │   └── Python.framework/          # 🐍 Python runtime
│   └── Resources/
│       ├── venv/                      # 📦 Virtual environment
│       │   ├── bin/python3           # ⚡ Python executable
│       │   └── lib/python3.12/       # 📚 Installed packages
│       ├── autocomplete.py           # 🤖 AI script
│       ├── minillm_export.pt         # 🧠 ML model
│       └── requirements.txt          # 📋 Dependencies
```

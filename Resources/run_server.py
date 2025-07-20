import os
from llama_cpp import Llama
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
from custom_chat_handler import CustomEmbeddingChatHandler
import uvicorn
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
import json

if __name__ == "__main__":
    # Путь к модели и LoRA-адаптерам
    model_dir = os.environ.get("MODEL_DIR", "./models")
    model_path = f"{model_dir}/gemma/gemma-3-4b-pt-q4_0.gguf"
    lora_dir = f"{model_dir}/lora"
    lora_adapters = [f"{lora_dir}/autocomplete_lora.gguf"]

    # Создаем настройки для модели, чтобы create_app мог ее найти
    settings = Settings(
        model=model_path,
        model_alias="gemma-3",
        n_gpu_layers=-1,
        embedding=True,
        lora_path=lora_adapters[0] if lora_adapters else None,
        n_ctx=4096,
        verbose=False  # Отключаем избыточные логи
    )

    # Создаем FastAPI приложение
    app = create_app(settings=settings)

    # Добавляем health check эндпоинт вручную
    @app.get("/health")
    def health():
        return {"status": "ok"}

    # Инициализация модели и обработчика для НАШЕГО кастомного эндпоинта
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        embedding=True,
        n_gpu_layers=-1,
        verbose=False,
        lora_adapters=[{"path": lora, "scale": 1.0} for lora in lora_adapters],
        cache=True
    )
    custom_handler = CustomEmbeddingChatHandler(llm)

    # Добавляем наш кастомный маршрут к созданному приложению
    @app.post("/custom_completion")
    async def custom_completion(request: Request):
        req_json = await request.json()
        stream = req_json.get("stream", False)

        # Используем наш кастомный обработчик
        if stream:
            def event_stream():
                # Handler возвращает генератор с dict объектами
                for res in custom_handler.complete(req_json):
                    delta = res.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                # Сигнал о завершении
                yield f"data: {json.dumps({'stop': True})}\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            # Синхронный режим — просто возвращаем JSON
            result = custom_handler.complete(req_json)
            return JSONResponse(result)

    # Запуск сервера
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=1
    )

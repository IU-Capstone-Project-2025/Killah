import sys
import time
import select
import json
import traceback
import requests
import os

# Максимальное количество токенов для генерации
MAX_TOKENS = 2048

def wait_for_server(server_url="http://localhost:8080/health", timeout=30):
    """Ждем пока сервер станет доступен"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(server_url)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def initialize_model():
    """Инициализируем подключение к серверу модели"""
    print("Initializing generation models...", file=sys.stderr, flush=True)
    print("Waiting for model server...", file=sys.stderr, flush=True)
    
    if wait_for_server():
        print("Model server is up.", file=sys.stderr, flush=True)
        print("Generation models initialized successfully.", file=sys.stderr, flush=True)
        return True
    else:
        print("Model server did not start within timeout.", file=sys.stderr, flush=True)
        return False

def get_lora_path(task_type="autocomplete"):
    """Получаем правильный путь к LoRA адаптеру для конкретной задачи"""
    model_dir = os.environ.get("MODEL_DIR")
    adapter_name = "autocomplete_lora_f16.gguf" # Default
    if task_type == "rewriting":
        adapter_name = "rewriting_lora_f16.gguf"
    elif task_type == "generation":
         adapter_name = "story_lora_f16.gguf"
    
    return os.path.join(model_dir, "lora", adapter_name)

def stream_generation(prompt_data: dict):
    """Стримим генерацию через HTTP запрос к серверу с LoRA адаптером и персонализацией"""
    
    global interrupted  # Добавляем доступ к глобальной переменной
    
    actual_prompt = prompt_data.get("prompt", "")
    task_type = prompt_data.get("task_type", "autocomplete")
    lora_path = prompt_data.get("lora_path", get_lora_path(task_type))
    context_embedding = prompt_data.get("context_embedding")
    temperature = prompt_data.get("temperature", 0.8)
    min_p = prompt_data.get("min_p", 0.1)

    # Логика из audio-ветки
    audio_embeddings = []
    if prompt_data.get("type") == "transcription":
        actual_prompt = prompt_data.get("text", "")
    elif prompt_data.get("type") == "projected_audio_embeds":
        audio_embeddings = prompt_data.get("embeddings", [[]])

    payload = {
        "prompt": actual_prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "min_p": min_p,
        "stream": True,
        "lora_path": lora_path,
        "audio_embeddings": audio_embeddings # Добавляем в payload
    }
    
    if context_embedding:
        payload["context_embedding"] = context_embedding

    try:
        print(f"🎯 [GENERATION] Task: {task_type}, LoRA: {os.path.basename(lora_path)}", file=sys.stderr, flush=True)
        if context_embedding:
            print(f"🎯 [GENERATION] With personalized context.", file=sys.stderr, flush=True)
        
        response = requests.post("http://127.0.0.1:8080/custom_completion", json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"Server error: {response.status_code} {response.text}", file=sys.stderr, flush=True)
            return
        
        buffer = ""
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if interrupted:  # Проверяем прерывание на каждом чанке
                print(f"[GENERATION] Interrupted during streaming.", file=sys.stderr, flush=True)
                return
            
            if not chunk:
                continue
            
            buffer += chunk
            while '\\n\\n' in buffer:
                message, buffer = buffer.split('\\n\\n', 1)
                if not message.startswith("data: "):
                    continue
                
                json_str = message[6:]
                try:
                    data = json.loads(json_str)
                    
                    if data.get("stop"):
                        return

                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        yield content

                except json.JSONDecodeError:
                    print(f"JSON decode error for chunk: {json_str}", file=sys.stderr, flush=True)
                    continue
                    
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error in stream_generation: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"FATAL Error in stream_generation: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    print("generation.py main loop started.", file=sys.stderr, flush=True)
    
    model_ready = initialize_model()
    if model_ready:
        print("READY", flush=True)

    current_request_data = None
    interrupted = False  # Глобальная переменная для прерывания
    
    while True:
        try:
            if not model_ready:
                print("Generation model not initialized. Exiting.", file=sys.stderr, flush=True)
                break

            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                line = sys.stdin.readline()
                if not line:
                    print("EOF received, exiting generation.py.", file=sys.stderr, flush=True)
                    break
                
                line = line.strip()
                if line.startswith("CMD:"):
                    command = line[4:]
                    if command == "ABORT":
                         print("[GENERATION] Abort command received.", file=sys.stderr, flush=True)
                         interrupted = True
                    else:
                        print(f"Unknown command: {command}", file=sys.stderr, flush=True)
                elif line:
                    try:
                        current_request_data = json.loads(line)
                        interrupted = True # Interrupt previous stream for the new one
                    except json.JSONDecodeError:
                         print(f"Error decoding JSON input: {line}", file=sys.stderr, flush=True)
                         current_request_data = None
                else:
                    current_request_data = None
                    interrupted = True

            if current_request_data:
                request_to_process = current_request_data
                current_request_data = None
                interrupted = False

                print("STREAM", flush=True)
                token_count = 0
                for token in stream_generation(request_to_process):
                    if interrupted:
                        print("[GENERATION] Stream interrupted.", file=sys.stderr, flush=True)
                        break
                    print(token, flush=True)
                    token_count += 1
                
                if not interrupted:
                    print("END", flush=True)
                
                print(f"[GENERATION] Stream finished. Total tokens: {token_count}", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"FATAL Error in generation main loop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            time.sleep(1)
            
    print("generation.py exited.", file=sys.stderr, flush=True) 
import sys
import time
import select
import json
import traceback
import requests
from main_llm import get_model_loader

# Максимальное количество токенов для автозаполнения
MAX_SUGGESTION_TOKENS = 100

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
    print("Initializing autocomplete models...", file=sys.stderr, flush=True)
    print("Waiting for model server...", file=sys.stderr, flush=True)
    
    if wait_for_server():
        print("Model server is up.", file=sys.stderr, flush=True)
        print("Autocomplete models initialized successfully.", file=sys.stderr, flush=True)
        return True
    else:
        print("Model server did not start within timeout.", file=sys.stderr, flush=True)
        return False

def stream_suggestions(prompt_text: str, temperature: float, lora_path: str, min_p: float = 0.1):
    """Стримим предложения через HTTP запрос к серверу с LoRA адаптером"""
    try:
        if prompt_text.startswith('{"'):
            prompt_data = json.loads(prompt_text)
            actual_prompt = prompt_data.get("prompt", "")
            lora_path = prompt_data.get("lora_path", lora_path)
        else:
            actual_prompt = prompt_text
        
        payload = {
            "prompt": actual_prompt,
            "messages": [{"role": "user", "content": actual_prompt}],
            "max_tokens": MAX_SUGGESTION_TOKENS,
            "temperature": temperature,
            "min_p": min_p,
            "stream": True,
            "lora_path": lora_path
        }
        
        print(f"Applying LoRA: {lora_path}", file=sys.stderr, flush=True)
        
        response = requests.post(
            "http://localhost:8080/custom_completion",
            json=payload,
            stream=True,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Server error: {response.status_code} {response.text}", file=sys.stderr, flush=True)
            return
        
        buffer = ""
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
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
                        print(f"Yielding token: '{content}'", file=sys.stderr, flush=True)
                        yield content

                except json.JSONDecodeError:
                    print(f"JSON decode error for chunk: {json_str}", file=sys.stderr, flush=True)
                    continue
                    
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error in stream_suggestions: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"FATAL Error in stream_suggestions: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    print("Autocomplete.py main loop started.", file=sys.stderr, flush=True)
    
    model_ready = initialize_model()
    if model_ready:
        print("READY", flush=True)

    current_prompt = None
    interrupted = False
    current_temperature = 0.8
    current_lora_adapter = "/Users/vladislavkalinichenko/Library/Containers/com.poinka.Killah-Prototype/Data/Library/Application Support/KillahPrototype/models/lora/autocomplete_lora_f16.gguf"
    
    while True:
        try:
            if not model_ready:
                print("Autocomplete model not initialized. Exiting.", file=sys.stderr, flush=True)
                break

            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                new_prompt_line = sys.stdin.readline()
                if not new_prompt_line:
                    print("EOF received, exiting autocomplete.py.", file=sys.stderr, flush=True)
                    break
                
                new_prompt = new_prompt_line.strip()
                if new_prompt.startswith("CMD:"):
                    command = new_prompt[4:]
                    if command == "INCREASE_TEMPERATURE":
                        current_temperature = min(current_temperature + 0.1, 2.0)
                        print(f"Temperature increased to {current_temperature}", file=sys.stderr, flush=True)
                    elif command == "DECREASE_TEMPERATURE":
                        current_temperature = max(current_temperature - 0.1, 0.1)
                        print(f"Temperature decreased to {current_temperature}", file=sys.stderr, flush=True)
                    else:
                        print(f"Unknown command: {command}", file=sys.stderr, flush=True)
                elif new_prompt:
                    current_prompt = new_prompt
                    interrupted = True
                else:
                    current_prompt = None
                    interrupted = True

            if current_prompt:
                prompt_to_process = current_prompt
                current_prompt = None
                interrupted = False

                if prompt_to_process:
                    print("STREAM", flush=True)
                    for token in stream_suggestions(prompt_to_process, current_temperature, current_lora_adapter, min_p=0.1):
                        if interrupted:
                            break
                        print(token, flush=True)
                    if not interrupted:
                        print("END", flush=True)
                else:
                    print("END", flush=True)
        
        except Exception as e:
            print(f"FATAL Error in autocomplete main loop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            time.sleep(5)
            
    print("Autocomplete.py exited.", file=sys.stderr, flush=True)

#!/usr/bin/env python3
"""
Custom Chat Handler with embedding injection support
Based on working llama-cpp-python 0.3.14 approach
"""
import os
import sys
import numpy as np
import ctypes
import threading
from llama_cpp import Llama, llama_cpp

_seq_counter = 0
_seq_counter_lock = threading.Lock()

def sample_min_p(logits, min_p):
    """
    Реализует min-p сэмплирование.
    https://arxiv.org/abs/2210.14141
    """
    if min_p == 0.0:
        return np.argmax(logits)
    
    probs = np.exp(logits) / np.sum(np.exp(logits))
    sorted_probs = np.sort(probs)[::-1]
    
    # Расчет k для min-p
    cumulative_probs = np.cumsum(sorted_probs)
    k = np.sum(cumulative_probs - sorted_probs > min_p)
    
    # Обрезаем логиты и сэмплируем
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Пересчитываем вероятности для top-k
    top_k_probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
    
    # Сэмплируем из обрезанного распределения
    sampled_index_in_top_k = np.random.choice(len(top_k_indices), p=top_k_probs)
    
    return top_k_indices[sampled_index_in_top_k]

class CustomEmbeddingChatHandler:
    def __init__(self, llm: Llama):
        self.base_llm = llm  # Базовая модель, загруженная один раз
        self.n_embd = llm.n_embd()
        self.model_path = None  # Путь к модели для создания новых экземпляров
        # Кэш для LoRA экземпляров, чтобы не создавать каждый раз заново
        self._lora_cache = {}

    def set_model_path(self, model_path: str):
        """Устанавливает путь к модели для создания LoRA экземпляров"""
        self.model_path = model_path

    def get_llm_with_lora(self, lora_path: str = None):
        """Возвращает экземпляр модели с нужным LoRA адаптером"""
        if not lora_path:
            return self.base_llm
        
        # Проверяем кэш
        if lora_path in self._lora_cache:
            return self._lora_cache[lora_path]
        
        # Создаем новый экземпляр с LoRA (переиспользует веса базовой модели)
        lora_llm = Llama(
            model_path=self.model_path,
            lora_adapters=[{"path": lora_path, "scale": 1.0}],
            n_ctx=4096,
            embedding=True,
            n_gpu_layers=-1,
            verbose=False,
            cache=True  # Переиспользует загруженные веса
        )
        # Кэшируем для повторного использования
        self._lora_cache[lora_path] = lora_llm
        return lora_llm

    def create_random_embedding(self):
        """Создает случайный эмбеддинг для тестирования."""
        return np.random.randn(self.n_embd).astype(np.float32)

    def create_batch_for_embeddings(self, embeddings, positions, seq_ids, logits_flags):
        """
        Создает batch для эмбеддингов используя рабочий подход
        """
        n_tokens = len(embeddings)
        
        # Подготавливаем массивы
        pos = np.array(positions, dtype=np.int32)
        n_seq_id = np.ones(n_tokens, dtype=np.int32)

        # Правильное создание массива указателей (int**)
        seq_id_arrays = [np.array([s_id], dtype=np.int32) for s_id in seq_ids]
        seq_id_pointers = (ctypes.POINTER(ctypes.c_int32) * n_tokens)()
        for i, arr in enumerate(seq_id_arrays):
            seq_id_pointers[i] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        
        logits = np.array(logits_flags, dtype=np.uint8)
        
        # Объединяем все эмбеддинги в один массив
        all_embeddings = np.concatenate(embeddings).astype(np.float32)
        
        # Создаем batch структуру как в рабочем примере
        null_int_ptr = ctypes.POINTER(ctypes.c_int32)()
        batch = llama_cpp.llama_batch(
            n_tokens=n_tokens,
            token=null_int_ptr,
            embd=all_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos=pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_seq_id=n_seq_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq_id=seq_id_pointers,
            logits=logits.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
            all_pos_0=0,
            all_pos_1=1,
            all_seq_id=0,
        )
        
        return batch

    def create_batch_for_tokens(self, tokens, positions, seq_ids, logits_flags):
        """
        Создает batch для токенов
        """
        n_tokens = len(tokens)
        
        # Подготавливаем массивы
        token_array = np.array(tokens, dtype=np.int32)

        # Валидация токенов: проверяем, что все id находятся в диапазоне словаря
        vocab_size = self.base_llm.n_vocab() # Используем base_llm для получения vocab_size
        invalid_tokens = [t for t in tokens if t < 0 or t >= vocab_size]
        if invalid_tokens:
            # Просто логируем ошибку, не падая
            print(f"[ERROR] Invalid token ids detected: {invalid_tokens[:10]}")
            # Можно заменить на более мягкую обработку, например, пропуск токенов
            # Для простоты пока оставим так, но без падения приложения
            return None # Сигнализируем об ошибке
        
        pos = np.array(positions, dtype=np.int32)
        n_seq_id = np.ones(n_tokens, dtype=np.int32)

        # Правильное создание массива указателей (int**)
        seq_id_arrays = [np.array([s_id], dtype=np.int32) for s_id in seq_ids]
        seq_id_pointers = (ctypes.POINTER(ctypes.c_int32) * n_tokens)()
        for i, arr in enumerate(seq_id_arrays):
            seq_id_pointers[i] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        logits = np.array(logits_flags, dtype=np.uint8)
        
        # Пустой массив эмбеддингов для токенов
        null_float_ptr = ctypes.POINTER(ctypes.c_float)()
        
        # Создаем batch структуру
        batch = llama_cpp.llama_batch(
            n_tokens=n_tokens,
            token=token_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            embd=null_float_ptr,
            pos=pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_seq_id=n_seq_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            seq_id=seq_id_pointers,
            logits=logits.ctypes.data_as(ctypes.POINTER(ctypes.c_byte)),
            all_pos_0=0,
            all_pos_1=1,
            all_seq_id=0,
        )
        
        return batch

    def complete(self, request, lora_path=None):
        """
        Обрабатывает запрос на генерацию текста с поддержкой min-p sampling, KV-caching и эмбеддингов.
        Запрос и ответ в формате словарей для совместимости с main_llm.py.
        """
        # Получаем нужный экземпляр модели (с LoRA или без)
        llm_instance = self.get_llm_with_lora(lora_path)
        
        return self._internal_complete(request, llm_instance, lora_path)
    
    def _internal_complete(self, request, llm_instance, lora_path=None):
        global _seq_counter
        # Извлекаем параметры
        max_tokens = request.get("max_tokens", 50)
        temperature = request.get("temperature", 0.8)
        min_p = request.get("min_p", 0.1)
        stream = request.get("stream", False)

        prompt_text = request.get("prompt", "")
        context_embedding = request.get("context_embedding")

        text_tokens = llm_instance.tokenize(prompt_text.encode("utf-8"), add_bos=False)
        
        with _seq_counter_lock:
            seq_id_this_request = _seq_counter
            _seq_counter += 1

        current_pos = 0
        all_embs = []

        if context_embedding:
            all_embs.append(np.array(context_embedding, dtype=np.float32))

        # Обработка эмбеддингов, если они есть
        if all_embs:
            # Логиты нужны для последнего эмбеддинга только если за ним не идут текстовые токены
            logits_flags = [False] * (len(all_embs) - 1) + [not text_tokens]
            
            emb_batch = self.create_batch_for_embeddings(
                embeddings=all_embs,
                positions=list(range(current_pos, current_pos + len(all_embs))),
                seq_ids=[seq_id_this_request] * len(all_embs),
                logits_flags=logits_flags
            )
            
            ret_emb = llama_cpp.llama_decode(llm_instance.ctx, emb_batch)
            if ret_emb != 0:
                raise RuntimeError(f"Ошибка декодирования эмбеддинга: {ret_emb}")
            
            current_pos += len(all_embs)

        # Обработка текстовых токенов
        if text_tokens:
            for i, token in enumerate(text_tokens):
                is_last_token = (i == len(text_tokens) - 1)
                
                token_batch = self.create_batch_for_tokens(
                    [token], 
                    [current_pos], 
                    [seq_id_this_request], 
                    [is_last_token]
                )
                
                if token_batch is None:
                    print(f"[WARNING] Skipping invalid token id at position {current_pos}")
                    continue

                ret_tok = llama_cpp.llama_decode(llm_instance.ctx, token_batch)
                if ret_tok != 0:
                    raise RuntimeError(f"Ошибка декодирования токена {i} ({token}): {ret_tok}")
                
                current_pos += 1

        # Если не было ни эмбеддингов, ни токенов, добавляем BOS токен для начала генерации
        if not all_embs and not text_tokens:
            bos_token = llm_instance.token_bos()
            token_batch = self.create_batch_for_tokens(
                [bos_token], 
                [0], 
                [seq_id_this_request], 
                [True]
            )
            ret_bos = llama_cpp.llama_decode(llm_instance.ctx, token_batch)
            if ret_bos != 0:
                raise RuntimeError(f"Ошибка декодирования BOS токена: {ret_bos}")

        # Получаем логиты для генерации
        # Индекс в батче, для которого нужны логиты, теперь 0, т.к. мы обрабатываем по одному
        logits_ptr = llama_cpp.llama_get_logits_ith(llm_instance.ctx, 0)
        
        if not logits_ptr:
             raise RuntimeError("Не удалось получить логиты от llama_get_logits_ith.")
        
        vocab_size = llm_instance.n_vocab()
        logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
        
        response_text = ""
        
        # Генерируем ответ с поддержкой streaming
        if stream:
            for i in range(max_tokens):
                token = sample_min_p(logits, min_p)
                
                decoded_text = llm_instance.detokenize([token]).decode("utf-8", "ignore")
                yield {"choices": [{"delta": {"content": decoded_text}}]}
                response_text += decoded_text
                
                # Проверяем токен остановки
                if token == llm_instance.token_eos():
                    break
                
                # Создаем batch для следующего токена
                next_token_batch = self.create_batch_for_tokens(
                    [token], 
                    [current_pos + i], 
                    [seq_id_this_request], 
                    [True]
                )

                if next_token_batch is None:
                    # Пропускаем невалидный токен
                    print(f"[WARNING] Skipping invalid generated token.")
                    continue

                ret = llama_cpp.llama_decode(llm_instance.ctx, next_token_batch)
                if ret != 0:
                    raise RuntimeError(f"Ошибка декодирования: {ret}")
                
                logits_ptr = llama_cpp.llama_get_logits_ith(llm_instance.ctx, 0)
                logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            
            # Отправляем сигнал завершения
            yield {"choices": [{"delta": {"content": ""}}], "stop": True}
        else:
            # Не-streaming генерация
            for i in range(max_tokens):
                token = sample_min_p(logits, min_p)
                response_text += llm_instance.detokenize([token]).decode("utf-8", "ignore")
                
                # Создаем batch для следующего токена
                next_token_batch = self.create_batch_for_tokens(
                    [token], 
                    [current_pos + i], 
                    [seq_id_this_request], 
                    [True]
                )

                if next_token_batch is None:
                    # Пропускаем невалидный токен
                    print(f"[WARNING] Skipping invalid generated token.")
                    continue

                ret = llama_cpp.llama_decode(llm_instance.ctx, next_token_batch)
                if ret != 0:
                    raise RuntimeError(f"Ошибка декодирования: {ret}")
                
                logits_ptr = llama_cpp.llama_get_logits_ith(llm_instance.ctx, 0)
                logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))

            return {"choices": [{"message": {"role": "assistant", "content": response_text}}]}

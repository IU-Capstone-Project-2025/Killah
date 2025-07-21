#!/usr/bin/env python3
import sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_attention(target, histories):
    """
    Вычисляет косинусное сходство между эмбеддингом target и каждым эмбеддингом в histories.
    """
    # Преобразуем входные данные в numpy массивы
    target_vector = np.array(target, dtype=np.float32).reshape(1, -1)
    histories_vectors = np.array(histories, dtype=np.float32)
    
    # Проверяем, что векторы не пустые и имеют одинаковую размерность
    if target_vector.shape[1] != histories_vectors.shape[1]:
        raise ValueError("Target and histories must have the same embedding dimension")
    if target_vector.size == 0 or histories_vectors.size == 0:
        raise ValueError("Empty vectors provided")
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(target_vector, histories_vectors)
    return similarities[0].tolist()

def normalize_attention_weights(similarities, top_n):
    """
    Нормализует значения внимания для топ-n документов так, чтобы минимальное было 0.1,
    максимальное 1, а остальные пропорционально между ними.
    Возвращает нормализованные веса и индексы топ-n документов.
    """
    if not similarities:
        return [], []

    # Сортируем индексы по убыванию сходства
    indexed_similarities = list(enumerate(similarities))
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Выбираем топ-n индексов и значений
    top_n = min(top_n, len(similarities))  # Убедимся, что не превышаем количество документов
    top_indices = [index for index, _ in indexed_similarities[:top_n]]
    top_similarities = [sim for _, sim in indexed_similarities[:top_n]]
    
    if not top_similarities:
        return [], []

    # Нормализация весов для топ-n документов
    min_sim = min(top_similarities)
    max_sim = max(top_similarities)
    
    # Если все значения одинаковые (например, все нули)
    if max_sim == min_sim:
        normalized = [1.0] * len(top_similarities)
    else:
        # Нормализация к диапазону [0.1, 1]
        normalized = [
            0.1 + 0.9 * (sim - min_sim) / (max_sim - min_sim)
            for sim in top_similarities
        ]
    
    return normalized, top_indices

if __name__ == "__main__":
    print("READY", flush=True)
    while True:
        try:
            input_data = input().strip()
            if input_data:
                data = json.loads(input_data)
                target = data['target']  # Ожидаем список чисел [float]
                histories = data['histories']  # Ожидаем список списков чисел [[float], [float], ...]
                top_n = data.get('top_n', 5)  # По умолчанию выбираем 3 документа
                similarities = cosine_similarity_attention(target, histories)
                normalized_weights, top_indices = normalize_attention_weights(similarities, top_n)
                result = {
                    "weights": normalized_weights,
                    "indices": top_indices
                }
                print(json.dumps(result), flush=True)
                print("END", flush=True)
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)

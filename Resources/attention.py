#!/usr/bin/env python3
import sys
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_attention(target, histories):
    """
    Computes cosine similarity between a target embedding and each embedding in histories.
    """
    target_vector = np.array(target, dtype=np.float32).reshape(1, -1)
    histories_vectors = np.array(histories, dtype=np.float32)

    if target_vector.shape[1] != histories_vectors.shape[1]:
        raise ValueError("Target and histories must have the same embedding dimension")
    if target_vector.size == 0 or histories_vectors.size == 0:
        raise ValueError("Empty vectors provided")

    similarities = cosine_similarity(target_vector, histories_vectors)
    return similarities[0].tolist()


def select_top_documents(embeddings, similarities, top_n=5, threshold=0.5):
    """
    Select the best documents using both top_n selection and threshold filtering.
    First selects top_n most similar documents, then applies threshold filtering.
    Returns selected embeddings and their normalized weights.
    """
    if not similarities or not embeddings:
        return [], []
    
    # Sort by similarity in descending order
    indexed_similarities = list(enumerate(zip(embeddings, similarities)))
    indexed_similarities.sort(key=lambda x: x[1][1], reverse=True)
    
    # Select top_n documents
    top_n = min(top_n, len(indexed_similarities))
    top_items = indexed_similarities[:top_n]
    
    # Extract embeddings and similarities for top documents
    top_embeddings = [item[1][0] for item in top_items]
    top_similarities = [item[1][1] for item in top_items]
    
    # Apply threshold filtering
    filtered_embeddings = []
    filtered_similarities = []
    for emb, sim in zip(top_embeddings, top_similarities):
        if sim > threshold:
            filtered_embeddings.append(emb)
            filtered_similarities.append(sim)
    
    if not filtered_embeddings:
        return [], []
    
    # Normalize weights to [0.1, 1.0] range for better combination
    min_sim = min(filtered_similarities)
    max_sim = max(filtered_similarities)
    
    if max_sim == min_sim:
        normalized_weights = [1.0] * len(filtered_similarities)
    else:
        normalized_weights = [
            0.1 + 0.9 * (sim - min_sim) / (max_sim - min_sim)
            for sim in filtered_similarities
        ]
    
    return filtered_embeddings, normalized_weights


def combine_embeddings_weighted(embeddings, weights):
    """
    Combines embeddings using weighted average.
    Returns None if no embeddings provided.
    """
    if not embeddings or not weights:
        return None
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    weights_array = np.array(weights, dtype=np.float32)
    
    # Weighted average
    weighted_sum = np.average(embeddings_array, axis=0, weights=weights_array)
    return weighted_sum.tolist()


if __name__ == "__main__":
    print("READY", flush=True)
    while True:
        try:
            raw = input().strip()
            if not raw:
                continue

            data = json.loads(raw)
            target = data["target"]
            histories = data["histories"]
            threshold = data.get("threshold", 0.5)
            top_n = data.get("top_n", 5)  # Default to top 5 documents
            
            # 1. Compute similarity scores
            similarities = cosine_similarity_attention(target, histories)
            
            # 2. Select best documents using top_n and threshold
            selected_embeddings, weights = select_top_documents(
                histories, similarities, top_n, threshold
            )
            
            # 3. Combine selected embeddings using weighted average
            combined = combine_embeddings_weighted(selected_embeddings, weights)
            
            # 4. Return combined embedding (or null if none selected)
            print(json.dumps({"combined_embedding": combined}), flush=True)
            print("END", flush=True)
            
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)

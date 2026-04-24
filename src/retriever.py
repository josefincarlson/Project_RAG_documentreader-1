# =========================
# retriever.py - Söker fram de mest relevanta textstyckena
# =========================
#
# Denna fil ansvarar för dokumentretrieval via embeddings och cosine similarity.
# Här beräknas likhet mellan användarens fråga och dokumentens textstycken,
# och de mest relevanta resultaten returneras med index, text och score.
# Om du vill ändra hur sökningen eller rankningen fungerar är detta rätt fil.


import numpy as np


# =========================
# Likhetsberäkning mellan embeddings
# =========================
#
# Denna del innehåller logik för att beräkna cosine similarity mellan en
# frågevektor och dokumentets embedding-matris. Resultatet används som grund
# för att avgöra vilka textstycken som är mest relevanta.

def cosine_similarity_batch(
    query_vec: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Beräknar cosine similarity mellan en fråge-vektor och en matris av embeddings.
    """
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(matrix))

    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)

    dot_products = matrix @ query_vec
    return dot_products / (matrix_norms * query_norm)


# =========================
# Semantisk sökning i dokumentchunks
# =========================
#
# Denna del ansvarar för att embedda användarens fråga, jämföra den med
# dokumentens embeddings och returnera de mest relevanta chunkarna utifrån
# similarity-score, tröskelvärde och top-k.

def semantic_search(
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    embed_query_func,
    k: int,
    threshold: float,
) -> list[dict]:
    """
    Söker fram de mest relevanta textstyckena utifrån en fråga.
    """
    if not query or not query.strip():
        raise ValueError("query är tom.")

    if not chunks:
        return []

    if chunk_embeddings is None:
        raise ValueError("chunk_embeddings saknas.")

    if not isinstance(chunk_embeddings, np.ndarray):
        raise TypeError("chunk_embeddings måste vara en numpy-array.")

    if chunk_embeddings.ndim != 2:
        raise ValueError("chunk_embeddings måste vara en 2-dimensionell numpy-array.")

    if embed_query_func is None:
        raise ValueError("embed_query_func saknas.")

    if k <= 0:
        raise ValueError("k måste vara större än 0.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold måste vara mellan 0 och 1.")
    
    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            f"Antal chunks ({len(chunks)}) matchar inte antal embeddingar ({len(chunk_embeddings)})."
        )

    query = query.strip()
    query_embedding = np.asarray(embed_query_func(query))

    if query_embedding.ndim != 1:
        raise ValueError("Query-embedding måste vara en 1-dimensionell vektor.")

    if chunk_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Dimensionen på query-embedding matchar inte dimensionen på chunk-embeddings."
        )
    
    scores = cosine_similarity_batch(query_embedding, chunk_embeddings)

    # Filtrera på threshold innan sortering
    valid_indices = np.where(scores >= threshold)[0]
    if len(valid_indices) == 0:
        return []

    # Sortera enbart de giltiga resultaten
    top_k_count = min(k, len(valid_indices))
    top_indices = valid_indices[np.argsort(scores[valid_indices])[::-1][:top_k_count]]

    return [
        {
            "index": int(i),
            "text": chunks[i],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]

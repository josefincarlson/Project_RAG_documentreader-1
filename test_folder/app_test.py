# =========================
# app_test.py — test av hela RAG-processen i en enkel script
# =========================

# Detta är en enkel testscript som kör hela RAG-processen i en icke-Streamlit-miljö.
# Användes under uppbyggnaden av appen för att snabbt testa att alla delar av RAG-processen fungerade som de skulle innan jag integrerade det i Streamlit-gränssnittet.
# Genom att ha denna testscript kan jag snabbt iterera på koden och felsöka eventuella problem i RAG-processen utan att behöva starta hela Streamlit-appen varje gång.


from src.loaders import load_document
from src.chunking import chunk_text_by_paragraphs
from src.embeddings import embed_texts, embed_query
from src.rag_pipeline import answer_query

if __name__ == "__main__":
    file_path = "test_docs/test1.txt"
    provider = "ollama"
    model_name = "llama3.1:8b"      # min lokala modell

    text = load_document(file_path)
    chunks = chunk_text_by_paragraphs(text)
    print(f"Antal chunks: {len(chunks)}")

    # Kommenterar bort så att jag inte ser alla chunkar i konsolen varje gång jag testar koden
    #for i, chunk in enumerate(chunks, start=1): 
    #    print(f"\n--- Chunk {i} ---")
    #    print(chunk)

    print("Skapar embeddings för chunks...")
    chunk_embeddings = embed_texts(chunks)                  # Skapa embeddings för alla chunks
    print("Skapar embeddings för chunks...")

    query = "Vad heter Ylvas systerdotter?"                                 # Testfråga   
    print(f"Fråga: {query}") 

    print("Kör answer_query...")
    answer, results, context = answer_query(
            query=query,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embed_query_func=embed_query,
            provider=provider,
            model_name=model_name,
            k=5
        )
    print("answer_query klar.")
      

    print(f"\nFråga: {query}")
    print(f"Provider: {provider}")
    print(f"Modell: {model_name}")
    print("\nTopp 5 mest relevanta chunks:\n")

    for rank, item in enumerate(results, start=1):
        print(f"--- Rank {rank} | Chunk {item['index']} | Score {item['score']:.4f} ---")
        print(item["text"])
        print()

    print("\nSvar från modellen:\n")
    print(answer)

    print("\nAlla scores:")
    for item in results:
        print(f"  Chunk {item['index']}: {item['score']:.4f} — {item['text'][:80]}...")

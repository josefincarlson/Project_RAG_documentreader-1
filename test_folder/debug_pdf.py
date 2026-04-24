# =========================
# debug_pdf.py — test av PDF-laddning och chunking
# =========================

# Användes vid utvecklingen av PDF-laddningen för att snabbt testa att PDF-filer laddades korrekt och chunkades på ett rimligt sätt.
# Genom att ha denna testscript kunde jag iterera snabbt på PDF-laddningen och chunking-logiken utan att behöva starta hela Streamlit-appen varje gång, vilket underlättade felsökning och utveckling av dessa delar av koden.
# I det här testet laddar jag en PDF-fil, chunkar den och skriver ut antalet chunkar samt de första 200 tecknen i de första 3 chunkarna för att verifiera att allt fungerar som det ska.


from src.loaders import load_document
from src.chunking import chunk_text_by_paragraphs

text = load_document("test_docs/Palme_1.pdf")
chunks = chunk_text_by_paragraphs(text)

print(f"Antal chunks: {len(chunks)}")
print("\nFörsta 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:200])
# =========================
# Återanvändbara UI-komponenter
# =========================
#
# Samlar Streamlit-komponenter som används på flera ställen i appen.

import re
import streamlit as st

from src.config import DOCUMENT_LOADING_SETTINGS, UI_SETTINGS


# =========================
# UI-konfiguration
# =========================
#
# Läser in och validerar configvärden som används av UI-komponenterna.

CATEGORY_OPTIONS = UI_SETTINGS.get("category_options")
SUPPORTED_EXTENSIONS = DOCUMENT_LOADING_SETTINGS.get("supported_extensions")
UPLOAD_HELP_TEXT = DOCUMENT_LOADING_SETTINGS.get("upload_help_text")

if not CATEGORY_OPTIONS:
    raise ValueError("UI_SETTINGS saknar 'category_options' i config.py.")

if not SUPPORTED_EXTENSIONS:
    raise ValueError(
        "DOCUMENT_LOADING_SETTINGS saknar 'supported_extensions' i config.py."
    )

if not UPLOAD_HELP_TEXT:
    raise ValueError(
        "DOCUMENT_LOADING_SETTINGS saknar 'upload_help_text' i config.py."
    )


# =========================
# Rating och feedback
# =========================
#
# Sparar användarfeedback per svar tillsammans med relevant metadata.

def render_rating_buttons(
        key_prefix: str,
        msg_index: int,
        query: str | None = None,
        answer: str | None = None,
        elapsed_time: float | None = None,
        model_name: str | None = None,
        provider: str | None = None,
    ) -> None:
    """
    Visar 👍/👎-knappar och sparar feedback per meddelandeindex i session_state.
    Varje assistantsvar kan därmed få en egen feedbackpost som går att uppdatera.
    """
    current_rating = st.session_state.ratings.get(msg_index)
    col_up, col_down, _ = st.columns([1, 1, 8])

    with col_up:
        if st.button(
            "👍✓" if current_rating == "bra" else "👍", 
            key=f"{key_prefix}_up_{msg_index}", 
            disabled=st.session_state.is_busy
        ):
            st.session_state.ratings[msg_index] = "bra"
            if query and answer:
                st.session_state.feedback[msg_index] = {
                    "query": query,
                    "answer": answer,
                    "rating": "bra",
                    "comment": "",
                    "issue_type": "",
                    "elapsed_time": elapsed_time,
                    "model_name": model_name,
                    "provider": provider,
                    "source_references": [],
                    "status": "new",
                    "user_identifier": "",
                    "user_display_name": "",
                }
                
            st.toast("Tack för din feedback! 👍")
            st.rerun()

    with col_down:
        if st.button(
            "👎✓" if current_rating == "dåligt" else "👎",
            key=f"{key_prefix}_down_{msg_index}",
            disabled=st.session_state.is_busy
        ):
            st.session_state.ratings[msg_index] = "dåligt"

            if query and answer:
                st.session_state.feedback[msg_index] = {
                    "query": query,
                    "answer": answer,
                    "rating": "dåligt",
                    "comment": "",
                    "issue_type": "",
                    "elapsed_time": elapsed_time,
                    "model_name": model_name,
                    "provider": provider,
                    "source_references": [],
                    "status": "new",
                    "user_identifier": "",
                    "user_display_name": "",
                }

            st.toast("Tack för din feedback! 👎")
            st.rerun()


# =========================
# Källvisning
# =========================
#
# Bygger och visar korta källreferenser med dokumentnamn, sida och stycke.

def extract_page_from_text(text: str) -> int | None:
    """
    Försöker läsa ut sidnummer ur chunktexten, till exempel [Sida 35].
    """
    if not text:
        return None

    match = re.search(r"\[Sida\s+(\d+)\]", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def build_source_references(
    results: list[dict],
    all_metadata: list[dict],
    max_sources: int = 3,
) -> list[str]:
    """
    Bygger korta och tydliga källreferenser från retrieval-resultaten.
    """
    if not results:
        return []

    if not isinstance(max_sources, int):
        raise TypeError("max_sources måste vara ett heltal.")

    if max_sources <= 0:
        raise ValueError("max_sources måste vara större än 0.")

    references = []
    seen = set()

    for result in results:
        if "index" not in result:
            continue

        chunk_index = result["index"]

        if not isinstance(chunk_index, int):
            continue

        if chunk_index < 0 or chunk_index >= len(all_metadata):
            continue

        metadata = all_metadata[chunk_index]
        doc_name = metadata.get("doc", "Okänt dokument")
        stycke = metadata.get("chunk_index")

        if isinstance(stycke, int):
            stycke += 1

        page_number = extract_page_from_text(result.get("text", ""))

        if page_number is not None and stycke is not None:
            ref = f"{doc_name} | Sida: {page_number} | Stycke: {stycke}"
        elif stycke is not None:
            ref = f"{doc_name} | Stycke: {stycke}"
        else:
            ref = doc_name

        if ref not in seen:
            references.append(ref)
            seen.add(ref)

        if len(references) >= max_sources:
            break

    return references


def render_source_references(source_references: list[str]) -> None:
    """
    Visar korta källreferenser, till exempel dokumentnamn och sida eller stycke.
    """
    if not source_references:
        return

    st.markdown("**Källor:**")
    for ref in source_references:
        st.caption(f"• {ref}")


# =========================
# Källchunks
# =========================
#
# Visar retrieval-resultat i en expanderbar lista med källa, score och kategori.

def render_source_chunks(
    results: list[dict],
    all_metadata: list[dict]
) -> None:
    """
    Visar källchunks i en expanderbar sektion.
    """
    with st.expander(f"📎 Källchunks ({len(results)} st)"):
        for rank, item in enumerate(results, start=1):
            if "index" not in item:
                raise ValueError("Ett sökresultat saknar 'index'.")

            if "text" not in item:
                raise ValueError("Ett sökresultat saknar 'text'.")

            if "score" not in item:
                raise ValueError("Ett sökresultat saknar 'score'.")

            if item["index"] < 0 or item["index"] >= len(all_metadata):
                raise ValueError("Sökresultatets index ligger utanför metadata-listan.")

            meta = all_metadata[item["index"]]
            page_number = extract_page_from_text(item["text"])

            if page_number is not None:
                source_caption = (
                    f"Rank {rank} | 📄 {meta['doc']} | "
                    f"Sida {page_number} | "
                    f"Stycke {meta['chunk_index'] + 1} | "
                    f"Score: {item['score']:.3f} | "
                    f"Kategori: {meta['category']}"
                )
            else:
                source_caption = (
                    f"Rank {rank} | 📄 {meta['doc']} | "
                    f"Stycke {meta['chunk_index'] + 1} | "
                    f"Score: {item['score']:.3f} | "
                    f"Kategori: {meta['category']}"
                )

            st.caption(source_caption)

            st.text(
                item["text"][:300] + "..."
                if len(item["text"]) > 300
                else item["text"]
            )

            st.divider()


# =========================
# Visning av chatthistorik
# =========================
#
# Denna del ansvarar för att rendera tidigare meddelanden i chatten,
# inklusive svarstid, modellinformation, källreferenser, rating-knappar
# och källchunks för assistantsvar.

def render_chat_history(all_metadata: list[dict]) -> None:
    """
    Visar alla tidigare meddelanden med rating, källreferenser och källchunks.
    """
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                render_source_references(msg.get("source_references", []))

            if msg["role"] == "assistant" and msg.get("elapsed_time") is not None:
                st.caption(
                    f"⏱️ Svarstid: {msg['elapsed_time']:.2f} sekunder | "
                    f"🤖 Provider: {msg.get('provider', 'okänd')} | "
                    f"Modell: {msg.get('model_name', 'okänd')}"
                )

            if msg["role"] == "assistant" and msg.get("results"):
                render_rating_buttons(
                    "hist",
                    msg_idx,
                    query=msg.get("query"), 
                    answer=msg.get("content"), 
                    elapsed_time=msg.get("elapsed_time"),
                    model_name=msg.get("model_name"), 
                    provider=msg.get("provider") 
                )
                render_source_chunks(msg["results"], all_metadata)


# =========================
# Visning av inlästa dokument
# =========================
#
# Denna del ansvarar för att visa dokument som redan har laddats in i appen.
# Här kan användaren se dokumentnamn, ändra kategori och ta bort dokument,
# vilket vid behov triggar ombyggnad av index och metadata.

def render_document_list(rebuild_fn) -> None:
    """
    Visar inlästa dokument med möjlighet att ta bort och ändra kategori.
    """
    if not st.session_state.documents:
        return

    st.subheader("📂 Inlästa dokument")
    docs_to_delete = []

    for fname, doc in st.session_state.documents.items():
        col_name, col_del = st.columns([4, 1])

        with col_name:
            st.markdown(f"**📄 {fname}**")

        with col_del:
            if st.button("🗑", key=f"del_{fname}", disabled=st.session_state.is_busy):
                docs_to_delete.append(fname)

        new_category = st.selectbox(
            f"Kategori för {fname}",
            options=CATEGORY_OPTIONS,
            index=CATEGORY_OPTIONS.index(doc["category"]) 
            if doc["category"] in CATEGORY_OPTIONS 
            else 0,
            key=f"category_{fname}",
            disabled=st.session_state.is_busy
        )

        if new_category != doc["category"]:
            st.session_state.documents[fname]["category"] = new_category
            for meta in st.session_state.documents[fname]["metadata"]:
                meta["category"] = new_category
            rebuild_fn()
            st.rerun()

        st.caption(f"Vald kategori: {doc['category']}")
        st.divider()

    for fname in docs_to_delete:
        del st.session_state.documents[fname]

    if docs_to_delete:
        rebuild_fn()
        st.rerun()


# =========================
# Uppladdning av dokument
# =========================
#
# Denna del innehåller UI för filuppladdning i sidopanelen. Här kan
# användaren välja filer, ange kategori per fil och starta uppladdning
# till appens dokumentlista.

def render_upload_section() -> tuple[list | None, bool]:
    """
    Visar filuppladdning och kategorival i sidopanelen.
    """
    st.subheader("📤 Ladda upp dokument")

    uploaded_files = st.file_uploader(
        "Välj en eller flera filer",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help=UPLOAD_HELP_TEXT,
        disabled=st.session_state.is_busy,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    st.caption(UPLOAD_HELP_TEXT)

    upload_clicked = False

    if uploaded_files:
        st.markdown("**Valda filer att ladda upp**")
        st.caption("Välj kategori per fil och klicka sedan på 'Ladda upp filer'.")

        current_names = {f.name for f in uploaded_files}
        st.session_state.pending_file_categories = {
            fname: cat 
            for fname, cat in st.session_state.pending_file_categories.items()
            if fname in current_names
        }

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if file_name not in st.session_state.pending_file_categories:
                st.session_state.pending_file_categories[file_name] = "Övrigt"

            st.markdown(f"**📄 {file_name}**")
            selected = st.selectbox(
                f"Kategori för {file_name}",
                options=CATEGORY_OPTIONS,
                index=CATEGORY_OPTIONS.index(
                    st.session_state.pending_file_categories[file_name]
                ),
                key=f"pending_category_{file_name}",
                disabled=st.session_state.is_busy
            )
            st.session_state.pending_file_categories[file_name] = selected
            st.divider()

        upload_clicked = st.button(
            "📥 Ladda upp filer", 
            disabled=st.session_state.is_busy)

    return uploaded_files, upload_clicked

# =========================
# Streamlit-appens huvudflöde
# =========================
#
# Binder ihop UI, session state och RAG-pipelinen.
# Här hanteras sidopanel, dokumentuppladdning och chattflöde.

import os
import tempfile
import time

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.loaders import load_document
from src.chunking import (
    chunk_text_by_paragraphs,
    filter_low_value_chunks,
    get_low_value_min_length_for_extension,
)
from src.embeddings import embed_texts, embed_query, get_embedding_metadata
from src.rag_pipeline import answer_query
from src.config_utils import (
    get_default_llm_provider, 
    get_default_llm_model, 
    get_allowed_llm_provider_names,
    get_llm_models_for_provider, 
    get_llm_provider_config, 
    validate_llm_policy, 
    get_active_embedding_config,
    get_security_mode,
    should_warn_on_cloud_usage,
    is_cloud_llm_allowed,
    is_cloud_embeddings_allowed,
)
from src.ui_components import (
    build_source_references,
    render_source_references,
    render_upload_section, 
    render_document_list, 
    render_chat_history, 
    render_rating_buttons, 
    render_source_chunks,
)


# =========================
# Miljö och API-nycklar
# =========================
#
# Små hjälpfunktioner för miljövariabler som påverkar appens beteende.

def gemini_api_key_exists() -> bool:
    """
    Returnerar True om Gemini-nyckeln finns i miljön och innehåller ett värde.
    Används för att avgöra om Gemini kan användas i UI-flödet.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    return bool(api_key and api_key.strip())


# =========================
# Session state
# =========================
#
# Sätter säkra standardvärden för appens tillstånd och lämnar befintliga värden orörda.

def initialize_session_state() -> None:
    """
    Initierar appens session_state första gången sidan laddas.
    Skapar bara nycklar som saknas, så att användarens pågående tillstånd bevaras.
    """
    defaults = {
        "documents": {},
        "vector_base": {
            "chunks": [],
            "embeddings": None,
            "metadata": [],
            "embedding_metadata": None,
        },
        "messages": [],
        "feedback": {},
        "ratings": {},
        "is_busy": False,
        "pending_query": None,
        "uploader_key": 0,
        "pending_file_categories": {},
        "upload_status_message": None,
        "pending_upload": [],
        "selected_provider": get_default_llm_provider(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = get_default_llm_model(
            st.session_state.selected_provider
        )

    if not isinstance(st.session_state.feedback, dict):
        st.session_state.feedback = {}

    if not isinstance(st.session_state.ratings, dict):
        st.session_state.ratings = {}


# =========================
# Grundinställningar
# =========================
#
# Sätter sidkonfiguration och startar session_state innan övrig logik körs.

st.set_page_config(page_title="Dokumentassistent", page_icon="📄", layout="centered")
st.title("📄 Intelligent dokumentassistent")
st.caption(
    "Ladda upp ett eller flera dokument och ställ frågor om innehållet. "
    "Lokalt via Ollama eller online via Gemini."
)
st.caption("Version: v1.2026.04.24")

initialize_session_state()


# =========================
# Vector base
# =========================
#
# Bygger upp och uppdaterar den sökbara dokumentbasen i sessionen.

def add_to_vector_base(
    new_chunks: list[str],
    new_embeddings,
    new_metadata: list[dict],
    embedding_metadata: dict,
) -> None:
    """
    Lägger till dokumentdata i aktuell vector_base.
    Stoppar blandning av embeddings från olika modeller i samma index.
    """
    vb = st.session_state.vector_base

    if vb["embedding_metadata"] is None:
        vb["embedding_metadata"] = embedding_metadata

    elif vb["embedding_metadata"] != embedding_metadata:
        raise ValueError(
            "Kan inte blanda embeddings från olika embedding-modeller i samma vector_base. "
            "Rensa dokumenten och bygg om indexet med den aktiva embedding-modellen."
        )

    vb["chunks"].extend(new_chunks)
    vb["metadata"].extend(new_metadata)

    if vb["embeddings"] is None:
        vb["embeddings"] = new_embeddings

    else:
        vb["embeddings"] = np.vstack([vb["embeddings"], new_embeddings])


def rebuild_vector_base() -> None:
    """
    Bygger om vector_base från de dokument som fortfarande finns kvar i sessionen.
    """
    st.session_state.vector_base = {
        "chunks": [],
        "embeddings": None,
        "metadata": [],
        "embedding_metadata": None,
    }

    for doc in st.session_state.documents.values():
        add_to_vector_base(
            doc["chunks"],
            doc["embeddings"],
            doc["metadata"],
            doc["embedding_metadata"],
        )


# =========================
# Hjälpfunktioner för embedding-konsistens
# =========================
#
# Denna del innehåller logik för att säkerställa att den aktiva
# embedding-konfigurationen matchar embeddings som redan finns i aktuell
# vector_base, så att olika embedding-modeller inte blandas i samma index.

def ensure_active_embedding_matches_vector_base() -> None:
    """
    Säkerställer att aktiv embedding-konfiguration matchar embeddings i aktuell vector_base.
    """
    vector_embedding_metadata = st.session_state.vector_base.get("embedding_metadata")

    # Om inga embeddings ännu finns i vector_base finns inget att jämföra.
    if vector_embedding_metadata is None:
        return

    current_embedding_metadata = get_embedding_metadata()

    if vector_embedding_metadata != current_embedding_metadata:
        raise ValueError(
            "Den aktiva embedding-modellen matchar inte dokumentens befintliga embeddings. "
            "Rensa dokumenten och ladda in dem igen med den aktuella embedding-modellen."
        )


# =========================
# Sidebar och appinställningar
# =========================
#
# Denna del ansvarar för sidopanelen med säkerhetsläge, val av provider
# och modell, varningar för molnanvändning, retrieval-inställningar samt
# uppladdnings- och dokumenthantering.

with st.sidebar:
    st.header("⚙️ Inställningar")
    security_mode = get_security_mode()

    if security_mode == "local_only":
        st.info("🔒 Säkerhetsläge: Endast lokalt")
    else:
        st.info("☁️🔒 Säkerhetsläge: Hybrid (lokalt + moln)")

    # Filtrerar providers utifrån aktiv säkerhetspolicy så att användaren
    # inte kan välja ett molnalternativ som är spärrat i aktuellt läge.
    provider_options = get_allowed_llm_provider_names()

    if st.session_state.selected_provider not in provider_options:
        st.session_state.selected_provider = provider_options[0]
        st.session_state.selected_model = get_default_llm_model(
            st.session_state.selected_provider
        )

    selected_provider = st.selectbox(
        "LLM-provider",
        options=provider_options,
        index=provider_options.index(st.session_state.selected_provider),
        disabled=st.session_state.is_busy,
    )

    if selected_provider != st.session_state.selected_provider:
        st.session_state.selected_provider = selected_provider
        st.session_state.selected_model = get_default_llm_model(selected_provider)

    model_options = get_llm_models_for_provider(st.session_state.selected_provider)

    if st.session_state.selected_model not in model_options:
        st.session_state.selected_model = get_default_llm_model(
            st.session_state.selected_provider
        )

    selected_model = st.selectbox(
        "Modell",
        options=model_options,
        index=model_options.index(st.session_state.selected_model),
        disabled=st.session_state.is_busy,
    )

    st.session_state.selected_model = selected_model

    provider = st.session_state.selected_provider
    model_name = st.session_state.selected_model

    provider_config = get_llm_provider_config(provider)
    embedding_config = get_active_embedding_config()

    if provider_config["execution_mode"] == "local":
        st.success("🔒 Lokalt LLM-läge — frågor och kontext stannar lokalt.")
    else:
        if (
            should_warn_on_cloud_usage()
            and provider_config.get("show_cloud_warning", False)
        ):
            st.error(
                "⚠️ VARNING: Du använder en molnbaserad LLM. "
                "Frågor och delar av dokumentinnehållet kan skickas till en extern tjänst. "
                "Använd endast detta läge för icke-konfidentiella dokument."
            )

        if not is_cloud_llm_allowed():
            st.error(
                f"⛔ Molnbaserade LLM-modeller är blockerade i säkerhetsläget '{security_mode}'."
            )

        if provider_config.get("requires_api_key", False):
            if gemini_api_key_exists():
                st.success("✅ GEMINI_API_KEY hittad.")
            else:
                st.error("❌ GEMINI_API_KEY saknas. Gemini kan inte användas i denna session.")

    if embedding_config["execution_mode"] == "cloud":
        if not is_cloud_embeddings_allowed():
            st.error(
                f"⛔ Molnbaserade embedding-modeller är blockerade i säkerhetsläget '{security_mode}'."
            )
        elif (
            should_warn_on_cloud_usage()
            and embedding_config.get("show_cloud_warning", False)
        ):
            st.warning(
                "⚠️ Aktiv embedding-modell är molnbaserad. "
                "Dokumenttext kan skickas till extern tjänst. "
                "Använd endast detta läge för icke-konfidentiella dokument."
            )
    else:
        st.caption(
            f"Embedding: {embedding_config['model_key']} ({embedding_config['model_name']})"
        )

    try:
        ensure_active_embedding_matches_vector_base()
    except ValueError as e:
        st.warning(f"⚠️ {e}")

    k = st.slider("Antal chunks att hämta", 2, 10, 2, disabled=st.session_state.is_busy)
    min_score = st.slider(
        "Minsta similarity-score",
        0.1,
        0.9,
        0.45,
        step=0.05,
        disabled=st.session_state.is_busy,
    )

    st.divider()    

    uploaded_files, upload_clicked = render_upload_section()

    st.divider()
    render_document_list(rebuild_vector_base)
    
    st.divider()
    st.caption("🔒 För konfidentiell data bör du använda helt lokalt läge.")


# =========================
# Filuppladdning och dokumentbearbetning
# =========================
#
# Denna del hanterar uppladdade filer, inläsning av dokument, chunking,
# filtrering, embeddings och uppdatering av sessionens vector_base och
# dokumentlista.

if upload_clicked and uploaded_files and not st.session_state.is_busy:
    st.session_state.is_busy = True
    st.session_state.pending_upload = uploaded_files
    st.rerun()

if (
    st.session_state.is_busy 
    and st.session_state.get("pending_upload") 
    and not st.session_state.pending_query
):
    uploaded_files = st.session_state.pending_upload
    new_files_processed = 0

    try:
        ensure_active_embedding_matches_vector_base()

        # En spinner visas tills alla nya dokument är färdigbearbetade
        with st.spinner("Bearbetar dokument, vänta..."):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name

                # Om filen redan finns inläst, hoppar vi över den
                if file_name in st.session_state.documents:
                    continue

                # Hämta vald kategori för just denna fil
                selected_category = st.session_state.pending_file_categories.get(
                    file_name, "Övrigt"
                )
                suffix = os.path.splitext(file_name)[1]
                tmp_path = None

                try:
                    # Skapar en temporär fil för att kunna läsa in dokumentet med våra befintliga funktioner som kräver en filväg. Den temporära filen tas bort direkt efter användning.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                   
                    text = load_document(tmp_path)                  # Läs in dokumentets text                 
                    raw_chunks = chunk_text_by_paragraphs(text)     # Delar upp texten i chunks/stycken

                    # Om inga chunks skapades, hoppar vi över dokumentet och visar en varning. Detta kan hända om dokumentet är tomt eller inte innehåller tillräckligt med text för att bilda ett stycke.
                    if not raw_chunks:
                        st.warning(
                            f"⚠️ {file_name} gav inga chunks och kunde därför inte läsas in."
                        )
                        continue

                    filter_min_length = get_low_value_min_length_for_extension(suffix)
                    chunks = raw_chunks

                    try:
                        filtered_chunks = filter_low_value_chunks(
                            raw_chunks,
                            min_length=filter_min_length,
                        )

                        # Om filtret tar bort för stor del av dokumentet är det bättre
                        # att behålla original-chunks än att göra retrieval nästan blind.
                        min_reasonable_chunk_count = max(5, int(len(raw_chunks) * 0.5))

                        if len(filtered_chunks) < min_reasonable_chunk_count:
                            st.info(
                                f"ℹ️ {file_name}: låg-värdesfiltret tog bort för stor del av dokumentet "
                                f"({len(filtered_chunks)}/{len(raw_chunks)} chunks kvar). "
                                "Original-chunks används i stället."
                            )
                        else:
                            chunks = filtered_chunks
                    except ValueError:
                        st.info(
                            f"ℹ️ {file_name}: filtret tog bort alla chunks. "
                            "Original-chunks används i stället."
                        )

                    embeddings = embed_texts(chunks)
                    embedding_metadata = get_embedding_metadata()
                    
                    chunk_metadata = [  # Metadata per chunk för källa och kategori
                        {
                            "doc": file_name,
                            "chunk_index": i,
                            "category": selected_category
                        }
                        for i in range(len(chunks))
                    ]

                    add_to_vector_base( # Lägger till de nya chunksen, embeddingarna och metadata i sessionens vector_base så att de blir sökbara direkt efter uppladdning. Vi säkerställer att samma embedding-modell används för alla dokument i indexet.
                        chunks, 
                        embeddings, 
                        chunk_metadata, 
                        embedding_metadata
                    )  
                    
                    # Sparar dokumentet i session_state
                    st.session_state.documents[file_name] = {
                        "chunks": chunks,
                        "embeddings": embeddings,
                        "metadata": chunk_metadata,
                        "category": selected_category, 
                        "embedding_metadata": embedding_metadata,
                    }

                    new_files_processed += 1

                except Exception as e:
                    st.error(f"Kunde inte läsa {file_name}: {e}")

                finally:
                    # Tar bort tempfilen efter användning
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # Efter att alla filer har bearbetats, uppdatera statusmeddelandet i session_state så att det visas i UI. Vi visar hur många nya dokument som laddades in eller om inga nya dokument laddades in.
        st.session_state.upload_status_message = (
            f"✓ {new_files_processed} nya dokument inlästa."
            if new_files_processed > 0
            else "Inga nya dokument laddades in. Filerna fanns redan eller gav inga chunks."
        )

        # Rensa tillfälliga kategorival
        st.session_state.pending_file_categories = {}

        # Nollställ file_uploader så att filerna försvinner visuellt därifrån
        st.session_state.uploader_key += 1

    except Exception as e:
        st.error(f"Kunde inte bearbeta uppladdningen: {e}")

    finally: 
        st.session_state.is_busy = False
        st.session_state.pending_upload = []

    st.rerun()


# =========================
# Statusmeddelanden och generell information
# =========================
#
# Denna del visar löpande status- och informationsmeddelanden i appen,
# till exempel begränsningar för PDF-stöd och resultat av uppladdning.

st.caption("📄 PDF-filer måste innehålla text — inskannade PDF:er (bilder) stöds inte ännu.")
st.caption("💡 Kategori och dokument kommer sparas till disk i nästa version.")

if st.session_state.upload_status_message:
    st.success(st.session_state.upload_status_message)
    st.session_state.upload_status_message = None


# =========================
# Huvudvy för inlästa dokument
# =========================
#
# Denna del visar dokumentöversikt och huvudgränssnittet när minst ett
# dokument finns inläst i appen.

if st.session_state.documents:
    # Hämta sökbas direkt från session_state — byggs inte om vid varje rendering
    all_chunks = st.session_state.vector_base["chunks"]
    all_embeddings = st.session_state.vector_base["embeddings"]
    all_metadata = st.session_state.vector_base["metadata"]

    st.subheader("📚 Dokumentöversikt")
    st.write(f"**Antal dokument:** {len(st.session_state.documents)}")
    st.write(f"**Antal chunks totalt:** {len(all_chunks)}")
    st.divider()

    if st.session_state.is_busy:
        st.warning("⏳ Systemet arbetar just nu. Inställningar är tillfälligt låsta.")
        st.caption(
            "Vänta tills bearbetningen är klar innan du ändrar inställningar eller dokument. "
            "Vill du avbryta direkt i utvecklingsläget kan du använda Stop-knappen högst upp "
            "i Streamlit eller ladda om sidan med F5."
        )
        

    # =========================
    # Chattflöde för frågor och svar
    # =========================
    #
    # Denna del hanterar chatthistorik, användarens nya frågor,
    # retrieval, svarsgenerering, källvisning och feedback på svar.

    st.subheader("💬 Ställ frågor om dokumenten")
    render_chat_history(all_metadata) #Denna funktion renderar hela chatthistoriken inklusive meddelanden, svarstider, modellnamn, rating-knappar och källchunks för varje assistantsvar. Den tar all_metadata som argument för att kunna visa relevant information om källchunks i historiken.

    query = st.chat_input(
        "Ställ en fråga om dokumenten...",
        disabled=st.session_state.is_busy
    )

    if query and not st.session_state.is_busy:
        st.session_state.pending_query = query
        st.session_state.is_busy = True
        st.rerun()

    if (
        st.session_state.is_busy
        and st.session_state.pending_query
        and not st.session_state.pending_upload
    ):
        query = st.session_state.pending_query
        st.session_state.messages.append({"role": "user", "content": query})
    
        with st.chat_message("user"):
            st.markdown(query)

        try:
            provider = st.session_state.selected_provider
            model_name = st.session_state.selected_model

            validate_llm_policy(provider=provider, model_name=model_name)
            ensure_active_embedding_matches_vector_base()

            with st.chat_message("assistant"):
                with st.spinner("Söker i dokumenten..."):
                    start_time = time.perf_counter()
                    answer, results, context = answer_query(
                        query=query,
                        chunks=all_chunks,
                        chunk_embeddings=all_embeddings,
                        all_metadata=all_metadata,
                        embed_query_func=embed_query,
                        provider=provider,
                        model_name=model_name,
                        k=k,
                        min_score=min_score,
                    )
                    elapsed_time = time.perf_counter() - start_time

                st.markdown(answer)

                source_references = build_source_references(
                    results, 
                    all_metadata, 
                    max_sources=3
                )
                render_source_references(source_references)

                st.caption(
                    f"⏱️ Svarstid: {elapsed_time:.2f} sekunder | "
                    f"🤖 Provider: {provider} | Modell: {model_name}"
                )

                msg_index = len(st.session_state.messages)
                render_rating_buttons(
                    "new",
                    msg_index,
                    query,
                    answer,
                    elapsed_time,
                    model_name,
                    provider,
                )

                if results:
                    render_source_chunks(results, all_metadata)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "results": results,
                    "source_references": source_references,
                    "elapsed_time": elapsed_time,
                    "provider": provider,
                    "model_name": model_name,
                    "query": query,
                }
            )

        except Exception as e:
            error_message = f"Kunde inte besvara frågan: {e}"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "results": [],
                    "source_references": [],
                    "elapsed_time": None,
                    "provider": st.session_state.selected_provider,
                    "model_name": st.session_state.selected_model,
                    "query": query,
                }
            )

            with st.chat_message("assistant"):
                st.error(error_message)

        finally:
            st.session_state.is_busy = False
            st.session_state.pending_query = None
            st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Rensa chatt", disabled=st.session_state.is_busy):
            st.session_state.messages = []
            st.session_state.ratings = {}
            st.session_state.feedback = {}
            st.rerun()


# =========================
# Tomt startläge
# =========================
#
# Denna del visar ett enkelt startmeddelande när inga dokument ännu har
# laddats upp i appen.

else:
    st.info("👈 Ladda upp ett eller flera dokument för att börja.")

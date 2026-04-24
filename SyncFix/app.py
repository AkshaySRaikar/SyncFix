# app.py
import streamlit as st
from PIL import Image
from retrieval.pipeline import RetrievalPipeline

st.set_page_config(page_title="SyncFix", page_icon="🔧", layout="wide")

@st.cache_resource
def load_pipeline():
    return RetrievalPipeline()

pipeline = load_pipeline()

st.title("🔧 SyncFix — Multimodal Technical Manual Search")
st.caption("Search across text, diagrams, and tutorial videos from your repair manuals.")

# ── Sidebar: settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results to retrieve (k)", 1, 10, 5)
    show_score = st.checkbox("Show similarity scores", True)
    st.divider()
    uploaded = st.file_uploader("Index a new PDF", type="pdf")
    if uploaded and st.button("Index PDF"):
        with st.spinner("Indexing…"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            from ingestion.index_pipeline import index_pdf
            index_pdf(tmp_path, "data/images")
            os.unlink(tmp_path)
        st.success("Indexed successfully!")

# ── Main: search
query = st.text_input("🔍 Search your manuals or ask a question…",
                       placeholder="e.g. How do I replace the carburetor needle?")

if query:
    with st.spinner("Retrieving…"):
        results = pipeline.retrieve(query, top_k=top_k)

    # ── Answer box
    st.subheader("💡 Answer")
    st.info(results["answer"])
    st.caption(f"Retrieval latency: **{results['latency_ms']} ms** · Top-{top_k} chunks retrieved")

    st.divider()
    st.subheader(f"📄 Top {top_k} Retrieved Chunks")

    for i, hit in enumerate(results["hits"]):
        with st.expander(f"Result {i+1} — Page {hit['page']} · {hit['pdf']}"
                          + (f" · Score: {hit['score']}" if show_score else "")):

            col_text, col_vis = st.columns([3, 2])

            # Text
            with col_text:
                st.markdown(f"**Excerpt:**")
                st.write(hit["text"])

            # Images
            with col_vis:
                if hit["images"]:
                    st.markdown("**Associated diagrams:**")
                    for img_path in hit["images"][:2]:  # show max 2
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.caption(f"Image not found: {img_path}")
                else:
                    st.caption("No diagrams on this page.")

                # YouTube embed
                if hit["yt_link"]:
                    st.markdown("**📺 Related video:**")
                    # Convert to embed URL
                    yt_embed = hit["yt_link"].replace("watch?v=", "embed/").replace("&t=", "?start=")
                    st.components.v1.iframe(yt_embed, height=200)
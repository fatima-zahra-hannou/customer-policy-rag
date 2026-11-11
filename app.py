"""Streamlit interface for the customer policy RAG assistant."""

import streamlit as st

from rag.generate import answer

st.set_page_config(page_title="Assistant CGV Sephora", page_icon="📋")


@st.cache_resource
def load_pipeline():
    """Trigger model loading once per session, not on every question."""
    from rag import generate  # noqa: F401

    return True


load_pipeline()

st.title("Assistant conseiller, conditions générales de vente")
st.caption("Pose une question comme le ferait un client, le système répond avec ses sources.")

question = st.text_input("Question du client")

if st.button("Chercher la réponse") and question:
    with st.spinner("Recherche en cours..."):
        result = answer(question)

    if result["abstained"]:
        st.warning(result["answer"])
    else:
        st.success(result["answer"])
        st.subheader("Sources citées")
        for s in result["sources"]:
            st.write(f"[{s['rank']}] page {s['page']} — {s['section']}")

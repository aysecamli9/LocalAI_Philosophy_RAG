import streamlit as st

from main import retrieve, generate_answer


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Philosophy RAG",
    page_icon="📚",
    layout="centered"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 Philosophy RAG")
st.caption("Local Retrieval-Augmented Generation for Philosophy")

st.divider()


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

query = st.text_area(
    "Ask a question about the philosophical texts:",
    placeholder="e.g. What is the state of nature like according to Hobbes?",
    height=100
)


# --------------------------------------------------
# ASK BUTTON
# --------------------------------------------------

if st.button("Ask", type="primary", use_container_width=True):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        # ------------------------------------------
        # RETRIEVAL
        # ------------------------------------------

        with st.spinner("Searching the knowledge base..."):

            try:
                retrieved_chunks, detected_philosophers = retrieve(query)

            except Exception as error:

                st.error(f"Retrieval failed: {error}")
                st.stop()


        # ------------------------------------------
        # GENERATION
        # ------------------------------------------

        with st.spinner("Generating answer..."):

            try:
                answer = generate_answer(
                    query,
                    retrieved_chunks,
                    detected_philosophers
                )

            except Exception as error:

                st.error(f"Generation failed: {error}")
                st.stop()


        # ------------------------------------------
        # ANSWER
        # ------------------------------------------

        st.subheader("Answer")

        st.write(answer)


        # ------------------------------------------
        # DETECTED PHILOSOPHERS
        # ------------------------------------------

        # if detected_philosophers:

            # st.caption(
            #     "Detected philosopher(s): "
            #     + ", ".join(detected_philosophers)
            # )


        # ------------------------------------------
        # SOURCES
        # ------------------------------------------

        st.subheader("Sources")

        for chunk in retrieved_chunks:

            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "?")
            similarity = chunk.get("similarity", 0)

            st.markdown(
                f"- **{source}** — Page {page} "
            )
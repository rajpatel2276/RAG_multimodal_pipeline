import streamlit as st
import requests
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/query")

st.set_page_config(page_title="CV Architecture Engine", layout="wide")
st.title("YOLO & CV Architecture Explorer")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Conversation History with Citations
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the assistant message has sources, render them
        if message.get("sources"):
            with st.expander("📚 View Citations & Visuals"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1} (Relevance: {source.get('score', 0):.2f})**")
                    if "source_image" in source and os.path.exists(source["source_image"]):
                        st.image(source["source_image"], use_container_width=True)
                    st.caption(source.get("snippet", ""))
                    st.divider()

if prompt := st.chat_input("Ask about CV architectures..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # RESTORED: The Loading Spinner
            with st.spinner("Processing vectors and synthesizing architecture..."):
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                payload = {"question": prompt, "history": history}
                
                response = requests.post(FASTAPI_URL, json=payload, timeout=120)
                response.raise_for_status() 
                
                data = response.json()
            
            answer = data.get("response") or data.get("answer") or data.get("text")
            sources = data.get("sources", [])
            
            if not answer:
                answer = f"Data received but key missing. Raw payload: {data}"
                
            message_placeholder.markdown(answer)
            
            # RESTORED: Real-time Citation Rendering
            if sources:
                with st.expander("📚 View Citations & Visuals"):
                    for idx, source in enumerate(sources):
                        st.markdown(f"**Source {idx+1} (Relevance: {source.get('score', 0):.2f})**")
                        # Look for local image paths embedded by LLaVA during ingestion
                        if "source_image" in source and os.path.exists(source["source_image"]):
                            st.image(source["source_image"], use_container_width=True)
                        st.caption(source.get("snippet", ""))
                        st.divider()

            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
        except requests.exceptions.RequestException as e:
            error_msg = f"**Backend Connection Failed:** `{str(e)}`"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
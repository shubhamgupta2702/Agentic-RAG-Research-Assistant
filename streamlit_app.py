import streamlit as st
import requests, os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 AI Research Assistant")
st.caption("Powered by LangGraph · Gemini · AstraDB · Tavily")


with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest Document"):
        with st.spinner("Ingesting…"):
            r = requests.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
            if r.status_code == 200:
                st.success(r.json()["message"])
            else:
                st.error(r.json().get("detail", "Error"))
    st.divider()
    st.markdown("**Routing logic**")
    st.markdown("📚 **RAG** — your uploaded docs")
    st.markdown("🌐 **Web Search** — live Tavily results")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
        if msg.get("route"):
            label = "📚 RAG" if msg["route"] == "rag" else "🌐 Web Search"
            st.caption(f"Routed via: {label}")


if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            try:
                r = requests.post(f"{API_BASE}/chat", json={"question": prompt})
                if r.status_code == 200:
                    data = r.json()
                    st.markdown(data["answer"])
                    if data.get("sources"):
                        with st.expander("📎 Sources"):
                            for s in data["sources"]:
                                st.markdown(f"- {s}")
                    label = "📚 RAG" if data.get("route") == "rag" else "🌐 Web Search"
                    st.caption(f"Routed via: {label}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "route": data.get("route", ""),
                    })
                else:
                    st.error(r.json().get("detail", "Error"))
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
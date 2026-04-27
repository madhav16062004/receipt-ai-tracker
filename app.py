"""
app.py — Streamlit UI for the AI Expense Chatbot
Features:
  • Sidebar dashboard with current-month spend
  • Tab 1: Upload receipt → OCR → LLM extraction → dual-DB save
  • Tab 2: Agentic chat interface (LangGraph + llama3.2)
"""

import streamlit as st
from backend.processor import extract_text_from_image, parse_with_ollama
from backend.memory import init_dbs, save_receipt_data, get_all_receipts, get_current_month_total, get_total_spend, reset_dbs
from backend.agent import build_agent, run_agent
import os
import tempfile

# ──────────────────────────────────────────────
# Page Config & DB Init
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Expense Chatbot",
    layout="wide",
)
init_dbs()

# ──────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

def _init_agent():
    """Build (or rebuild) the LangGraph agent."""
    try:
        st.session_state.agent = build_agent()
    except Exception as e:
        st.session_state.agent = None
        st.error(f"Failed to initialize agent: {e}")

if "agent" not in st.session_state:
    with st.spinner("Initializing AI Agent..."):
        _init_agent()

# ──────────────────────────────────────────────
# Sidebar — Dashboard
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("Dashboard")
    st.divider()

    month_total = get_current_month_total()
    all_time_total = get_total_spend()

    m1, m2 = st.columns(2)
    m1.metric(label="This Month", value=f"₹ {month_total:,.2f}")
    m2.metric(label="All Time", value=f"₹ {all_time_total:,.2f}")

    st.divider()

    df = get_all_receipts()
    if not df.empty:
        st.subheader("Spending by Category")
        chart_data = df.groupby("category")["total"].sum()
        st.bar_chart(chart_data)

        st.subheader(f"Total Receipts: {len(df)}")
    else:
        st.info("No receipts yet. Upload one to get started!")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(" Reset Agent", use_container_width=True):
            _init_agent()
            st.toast("Agent rebuilt!")
    with col_b:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    if st.button("Reset Database", use_container_width=True):
        reset_dbs()
        st.session_state.messages = []
        st.toast("Databases completely wiped!")
        st.rerun()

    st.caption("Powered by EasyOCR • Gemini • LangGraph")


# ──────────────────────────────────────────────
# Main Area — Tabs
# ──────────────────────────────────────────────
st.title(" AI Expense Chatbot")

tab_upload, tab_chat, tab_history = st.tabs(["Upload Receipt", " Chat", " History"])


# ──────────────────────────────────────────────
# Tab 1: Upload Receipt
# ──────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a Receipt Image")

    uploaded_file = st.file_uploader(
        "Drag & drop or browse", type=["png", "jpg", "jpeg"], key="receipt_upload"
    )

    if uploaded_file:
        # Show image preview
        col_img, col_data = st.columns([1, 2])

        with col_img:
            st.image(uploaded_file, caption="Uploaded Receipt", use_container_width=True)

        # Save to a temp file for EasyOCR
        tmp_path = os.path.join(tempfile.gettempdir(), "receipt_upload.jpg")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with col_data:
            # Initialize cache for this specific file
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if "current_file_key" not in st.session_state or st.session_state.current_file_key != file_key:
                # New file uploaded -> Run OCR and LLM
                st.session_state.current_file_key = file_key
                
                with st.spinner(" Step 1: Reading text with EasyOCR..."):
                    raw_text = extract_text_from_image(tmp_path)
                
                if raw_text:
                    with st.spinner("Step 2: AI Analysis with Ollama..."):
                        structured_data = parse_with_ollama(raw_text)
                else:
                    structured_data = None

                st.session_state.raw_text = raw_text
                st.session_state.structured_data = structured_data

            # Retrieve from cache
            raw_text = st.session_state.raw_text
            structured_data = st.session_state.structured_data

            if not raw_text:
                st.error("Could not read any text from the image. Try a clearer photo.")
            else:
                st.text_area("Raw OCR Text", raw_text, height=100, disabled=True)

                if structured_data:
                    st.success(" Extraction Complete!")

                    # Editable fields
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        merchant = st.text_input("Merchant", structured_data.get("merchant", ""))
                        date = st.text_input("Date (YYYY-MM-DD)", structured_data.get("date", ""))
                    with c2:
                        total = st.number_input(
                            "Total Amount",
                            value=float(structured_data.get("total", 0.0)),
                            format="%.2f",
                        )
                        currency = st.text_input("Currency", structured_data.get("currency", "₹"))
                    with c3:
                        categories = ["Food", "Transport", "Shopping", "Utilities", "Other"]
                        default_cat = structured_data.get("category", "Other")
                        cat_index = categories.index(default_cat) if default_cat in categories else 4
                        category = st.selectbox("Category", categories, index=cat_index)

                    if st.button(" Save to Database", type="primary"):
                        final_data = {
                            "merchant": merchant,
                            "date": date,
                            "total": total,
                            "currency": currency,
                            "category": category,
                        }
                        save_receipt_data(raw_text, final_data)
                        st.toast(f"Saved {currency} {total:.2f} at {merchant}!")
                else:
                    st.error(" AI failed to parse the receipt. Please try a clearer photo.")


# ──────────────────────────────────────────────
# Tab 2: Chat Interface
# ──────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask me anything about your expenses")
    st.caption("I can query your database, search receipt details, and give you spending insights.")

    # Create a scrollable container for chat messages so the input stays anchored
    chat_container = st.container(height=500)

    # Render chat history
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("e.g. How much did I spend on Food this month?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate agent response
            with st.chat_message("assistant"):
                if st.session_state.agent is None:
                    response = " Agent is not initialized. Make sure your GOOGLE_API_KEY is set in the .env file."
                else:
                    with st.spinner(" Thinking..."):
                        response = run_agent(
                            st.session_state.agent,
                            user_input,
                            st.session_state.messages[:-1],  # history excluding current msg
                        )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ──────────────────────────────────────────────
# Tab 3: History & Analytics
# ──────────────────────────────────────────────
with tab_history:
    st.subheader("All Receipts")
    df = get_all_receipts()
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No receipts found. Upload your first receipt to get started!")
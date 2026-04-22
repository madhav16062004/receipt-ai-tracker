import streamlit as st
from core.processor import extract_text_from_image, parse_with_ollama
from core.database import init_db, save_receipt, get_all_receipts
import os

# Initialize App
st.set_page_config(page_title="AI Receipt Tracker", layout="wide")
init_db()

st.title("🧾 AI Expense Tracker (EasyOCR + Ollama)")

tab1, tab2 = st.tabs(["Upload Receipt", "History & Analytics"])

with tab1:
    uploaded_file = st.file_uploader("Scan a receipt", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        # Save temp file
        with open("temp_receipt.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Step 1: Reading text with EasyOCR..."):
            raw_text = extract_text_from_image("temp_receipt.jpg")
            st.text_area("Raw Text Detected:", raw_text, height=100)

        with st.spinner("Step 2: AI Analysis with Ollama..."):
            structured_data = parse_with_ollama(raw_text)
            
            if structured_data:
                st.success("Extraction Complete!")
                # Create editable fields
                col1, col2, col3 = st.columns(3) # Changed to 3 columns
                with col1:
                    merchant = st.text_input("Merchant", structured_data.get('merchant'))
                    date = st.text_input("Date", structured_data.get('date'))
                with col2:
                    total = st.number_input("Total Amount", value=float(structured_data.get('total', 0.0)))
                    currency = st.text_input("Currency", structured_data.get('currency', '$')) # Added this
                with col3:
                    category = st.selectbox("Category", ["Food", "Transport", "Shopping", "Utilities", "Other"])

                if st.button("Save to Database"):
                    save_receipt({
                        "merchant": merchant,
                        "date": date,
                        "total": total,
                        "currency": currency, # Pass it to the database function
                        "category": category
                    })
                    st.toast(f"Saved {currency}{total} successfully!")
            else:
                st.error("AI failed to parse the text. Try a clearer photo.")

with tab2:
    df = get_all_receipts()
    if not df.empty:
        st.subheader("Recent Expenses")
        st.dataframe(df, use_container_width=True)
        
        # Simple Chart
        st.subheader("Spending by Category")
        chart_data = df.groupby('category')['total'].sum()
        st.bar_chart(chart_data)
    else:
        st.info("No receipts found in database.")
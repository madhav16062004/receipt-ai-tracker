# Receipt AI Tracker 

Welcome to the **Receipt AI Tracker**! This project is a smart expense tracking application that allows you to upload photos of your receipts, automatically extract and structure the details using OCR and local AI models, and chat with an intelligent agent to analyze your spending habits.

## Features

- **Automated Receipt Processing:** Upload images of your receipts and let the app extract merchant, date, total, currency, and category automatically.
- **Conversational Analytics:** Chat with an AI assistant to ask natural language questions about your expenses (e.g., "How much did I spend on food this month?", "What was my highest expense?").
- **Multi-Currency Support:** Handles purchases in multiple currencies ($, €, £, ₹) and normalizes them for fair comparison.
- **Dual-Database System:** Uses SQLite for structured data tracking and ChromaDB for semantic vector searches across raw receipt text.
- **Sleek UI:** A responsive, dark-mode glassmorphism interface built with vanilla HTML, CSS, and JS.

---

##  Architecture

The project is structured with a distinct backend and frontend:

- **Backend (FastAPI):** Exposes REST APIs for uploading/processing receipts, querying statistics, and interacting with the AI agent. The endpoints use standard synchronous threading to prevent blocking the event loop while heavy operations run.
- **Frontend (Vanilla HTML/CSS/JS):** A lightweight, premium-looking interface without the overhead of heavy frameworks like React or Vue.
- **Data Processing (EasyOCR & Ollama):** 
  - `EasyOCR` reads the raw text from the uploaded image quickly.
  - `Ollama` (running a fast local model like `qwen2.5:3b-instruct`) parses the raw text into structured JSON.
- **AI Agent (LangGraph & Gemini API):** A ReAct agent powered by Gemini that decides when to run SQL queries against the structured database or do a semantic search in ChromaDB to answer your questions.
- **Database:**
  - `SQLite` stores the structured details (Date, Merchant, Total, Currency, Category).
  - `ChromaDB` stores embeddings of the raw receipt text (using `nomic-embed-text`) for fuzzy searching.

---

## 🛠️ Installation & Setup

Follow these steps to run the application on your local machine.

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed on your machine.

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd receipt-ai-tracker
```

### 3. Set Up Python Environment
Create a virtual environment and activate it:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 5. Start Ollama and Pull Required Models
Make sure the Ollama application is running on your machine, then open a terminal and pull the models used by the system:
```bash
ollama run qwen2.5:3b-instruct
ollama pull nomic-embed-text
```

### 6. Configure Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

##  Usage

### 1. Start the Server
With your virtual environment activated, start the FastAPI backend using Uvicorn:
```bash
python -m uvicorn api:app --reload --port 8000
```

### 2. Access the Application
Open your web browser and navigate to:
```
http://localhost:8000
```

### 3. Using the App
- **Upload Receipt Tab:** Drag and drop an image of a receipt. The system will perform OCR and use AI to extract the details. Review the fields and click "Save to Database".
- **Chat Tab:** Talk to the AI! Ask analytical questions like "What did I spend the most on?" or "Show me my transport expenses". The AI agent will dynamically query the database and present the answer.
- **History Tab:** View a clean table of all the receipts you have saved so far.
- **Dashboard (Left Sidebar):** Instantly see your spending totals and category breakdowns.

---

##  Notes
- The initial run of the OCR engine (`EasyOCR`) might take slightly longer as it downloads the detection models for the first time.
- If the chat agent gets confused or you wish to start fresh, you can hit the "Reset Agent" or "Clear Chat" buttons in the sidebar.

"""
agent.py - LangGraph agentic chatbot.

Defines tools for expense analytics and receipt search, then wires them
into a LangGraph ReAct agent powered by Gemini API.

Tools are intentionally "dumb" data fetchers — all reasoning, filtering,
and interpretation is done by the LLM.
"""

import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from backend.memory import SQLITE_PATH, convert_to_inr, execute_sql, semantic_search

load_dotenv()


# ──────────────────────────────────────────────
# Tools — simple data access, no NLP logic
# ──────────────────────────────────────────────

@tool
def query_sql_database(sql_query: str) -> str:
    """Execute a SQL SELECT query against the receipts database and return results.

    Table schema:
        receipts (
            id      INTEGER PRIMARY KEY,
            date    TEXT,        -- format: YYYY-MM-DD
            merchant TEXT,
            total   REAL,
            currency TEXT,       -- e.g. '$', '₹', '€', '£'
            category TEXT        -- one of: Food, Transport, Shopping, Utilities, Other
        )

    Only SELECT queries are allowed. Write your own SQL based on the user's question.
    """
    return execute_sql(sql_query)


@tool
def get_all_receipts_summary() -> str:
    """Fetch every receipt from the database with totals converted to INR.

    Use this to get a full overview of all spending data. Returns each receipt
    with merchant, amount in original currency, equivalent INR amount, category,
    and date, plus a grand total.
    """
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        c = conn.cursor()
        c.execute("SELECT merchant, total, currency, category, date FROM receipts ORDER BY date DESC")
        rows = c.fetchall()
        conn.close()

        if not rows:
            return "No receipts found in the database."

        currency_totals = {}
        total_inr = 0.0
        breakdown = []

        for merchant, total, currency, category, date in rows:
            curr = (currency or "INR").strip()
            inr_amount = convert_to_inr(total, curr)
            total_inr += inr_amount
            currency_totals[curr] = currency_totals.get(curr, 0.0) + total
            breakdown.append(
                f"  - {merchant}: {curr}{total:.2f} (= INR {inr_amount:,.2f}) [{category}, {date}]"
            )

        result = f"GRAND TOTAL (INR): INR {total_inr:,.2f}\n"
        result += "ORIGINAL CURRENCY TOTALS:\n"
        for curr, amt in currency_totals.items():
            result += f"  {curr}{amt:.2f}\n"
        result += f"Number of receipts: {len(rows)}\n"
        result += "BREAKDOWN:\n" + "\n".join(breakdown)
        return result
    except Exception as e:
        return f"Error: {e}"


@tool
def search_receipt_context(query: str) -> str:
    """Search receipt contents using semantic similarity (vector search).

    Use this for fuzzy or descriptive questions about specific items,
    products, or receipt text that wouldn't be captured in the structured fields.
    """
    return semantic_search(query, n_results=5)


TOOLS = [query_sql_database, get_all_receipts_summary, search_receipt_context]


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build the system prompt with the current date injected."""
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""You are an intelligent AI expense assistant. You help users understand their spending by querying their receipt database.

TODAY'S DATE: {today}

You have three tools:

1. query_sql_database
   Write and execute SQL SELECT queries against the receipts table.
   Table schema: receipts (id INTEGER, date TEXT, merchant TEXT, total REAL, currency TEXT, category TEXT)
   Dates are in YYYY-MM-DD format. Currency values include '$', '₹', '€', '£'.
   Categories are: Food, Transport, Shopping, Utilities, Other.
   Use this for filtering by date, category, merchant, or counting receipts.

2. get_all_receipts_summary
   Fetches ALL receipts with per-receipt breakdown and each amount converted to INR.
   Use this when the user wants a broad spending overview or you need to see everything.

3. search_receipt_context
   Semantic search over raw receipt text. Use this for fuzzy questions about specific items or products.

CRITICAL — Multi-Currency Warning:
- The 'total' column stores amounts in DIFFERENT currencies (USD, INR, EUR, etc.).
- You CANNOT compare or rank totals using SQL alone, because $235 is worth more than ₹1392.
- For ANY question involving comparison, ranking, "most", "least", "highest", "biggest", or aggregation across receipts with mixed currencies, you MUST use get_all_receipts_summary. That tool converts every amount to INR so you can compare fairly.
- Only use query_sql_database for comparisons when all receipts share the same currency, or for non-monetary queries (counts, dates, categories, etc.).

Rules:
- Always use a tool to get data before answering. Never guess or make up numbers.
- YOU decide which tool to use and what parameters to pass.
- Never show SQL queries, tool names, or internal reasoning to the user.
- Present answers in a clear, conversational format. Use the exact numbers from the tools.
- When showing monetary amounts, prefer INR equivalents unless the user asks for original currency.
"""


# ──────────────────────────────────────────────
# Agent Builder & Runner
# ──────────────────────────────────────────────

def build_agent():
    """Build and return a compiled LangGraph ReAct agent."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=api_key,
        temperature=0,
    )
    return create_react_agent(llm, TOOLS, prompt=_build_system_prompt())


def run_agent(agent, user_message: str, chat_history: list) -> str:
    """Invoke the agent with the user's message and full chat history.

    Every question goes through the agent — no shortcuts or bypasses.
    The LLM decides which tools to call and how to interpret results.
    """
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))

    try:
        result = agent.invoke({"messages": messages})
        final_messages = result.get("messages", [])
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                content = msg.content
                # Gemini may return content as a list of dicts instead of a string
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(part["text"])
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "\n".join(parts)
                return content
        return "I processed your request but couldn't generate a response. Please try again."
    except Exception as e:
        return f"Agent error: {e}\n\nMake sure your GOOGLE_API_KEY is set correctly in the .env file."

"""
agent.py — LangGraph Agentic Chatbot
Defines two tools (SQL query + semantic search) and wires them
into a LangGraph ReAct agent powered by llama3.2 via Ollama.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from core.memory import execute_sql, semantic_search, convert_to_inr, SQLITE_PATH
import sqlite3


# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────

@tool
def query_sql_database(question: str) -> str:
    """Use this tool when the user asks an ANALYTICAL question about their expenses,
    such as totals, averages, counts, comparisons, or filtering by date/category/merchant.
    
    You must convert the user's natural-language question into a valid SQLite SELECT query.
    
    The database has a single table called 'receipts' with these columns:
        id       INTEGER PRIMARY KEY
        date     TEXT   (format: YYYY-MM-DD)
        merchant TEXT
        total    REAL
        currency TEXT
        category TEXT   (one of: Food, Transport, Shopping, Utilities, Other)

    Write the SQL query and pass it as the 'question' argument.
    Example: SELECT SUM(total) FROM receipts WHERE category = 'Food'
    """
    return execute_sql(question)


@tool
def get_spending_summary(filter_sql: str) -> str:
    """Use this tool when the user asks about TOTAL SPENDING or HOW MUCH they spent.
    This tool handles currency conversion to INR automatically.
    
    Pass a SQL WHERE clause (without the WHERE keyword) to filter receipts.
    Examples:
        - All spending: pass "1=1"
        - Food spending: pass "category = 'Food'"
        - This month: pass "date LIKE '2026-04%'"
        - Year 2018: pass "date LIKE '2018%'"
        - At a merchant: pass "merchant LIKE '%walmart%'"
    
    Returns totals in both original currency and INR (₹).
    """
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        c = conn.cursor()
        c.execute(f"SELECT merchant, total, currency, category, date FROM receipts WHERE {filter_sql}")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return "No matching receipts found."
        
        # Group by currency for accurate original-currency totals
        currency_totals = {}
        total_inr = 0.0
        breakdown = []
        
        for merchant, total, currency, category, date in rows:
            curr = (currency or "INR").strip()
            inr_amount = convert_to_inr(total, curr)
            total_inr += inr_amount
            currency_totals[curr] = currency_totals.get(curr, 0.0) + total
            breakdown.append(
                f"  - {merchant}: {curr}{total:.2f} (= ₹{inr_amount:,.2f}) [{category}, {date}]"
            )
        
        # Build summary
        result = f"GRAND TOTAL (INR): ₹{total_inr:,.2f}\n"
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
    """Use this tool when the user asks a SEMANTIC or DESCRIPTIVE question about
    their receipts, such as 'What did I buy at Walmart?', 'Show me grocery receipts',
    or any question about specific items or receipt contents.
    
    Pass the user's query as-is; the tool will search the vector database
    for the most relevant receipt texts.
    """
    return semantic_search(query, n_results=5)


# ──────────────────────────────────────────────
# Agent Setup
# ──────────────────────────────────────────────

TOOLS = [query_sql_database, get_spending_summary, search_receipt_context]


def _build_system_prompt() -> str:
    """Build the system prompt with the current date injected."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.now().strftime("%Y-%m")

    return f"""You are an intelligent AI expense assistant. You help users understand their spending by querying their receipt database.

TODAY'S DATE: {today}
CURRENT MONTH PREFIX: '{current_month}%'

You have three tools:

1. **get_spending_summary** — PREFERRED tool for any "how much did I spend" question.
   It automatically converts all currencies to INR (₹).
   Pass a SQL WHERE clause (without WHERE keyword) as the argument.
   Examples:
     - All spending: "1=1"
     - Food only: "category = 'Food'"
     - This month: "date LIKE '{current_month}%'"

2. **query_sql_database** — For detailed analytical queries (listing receipts, counting, grouping).
   Pass a full SQLite SELECT query.
   Table: receipts (id, date TEXT, merchant TEXT, total REAL, currency TEXT, category TEXT).

3. **search_receipt_context** — For semantic questions about receipt contents/items.

STRICT RESPONSE RULES:
- ALWAYS use a tool before answering. NEVER guess or make up numbers.
- Do NOT ask for clarification when you can infer from context.
- NEVER show SQL, tool names, or internal reasoning in your response.
- NEVER say "Here's the query", "Running this query", or "Let me use the tool".
- Give a DIRECT conversational answer.
- CRITICAL: Use the EXACT numbers returned by the tool. NEVER do math yourself. NEVER calculate or convert numbers yourself.
- The tool returns both "GRAND TOTAL (INR)" and "ORIGINAL CURRENCY TOTALS". Use whichever the user asks for.
- If user asks "in dollars" or "in USD", use the dollar amount from ORIGINAL CURRENCY TOTALS.
- By default, present in INR (₹) using the GRAND TOTAL value.
- Include a brief breakdown with bullet points when there are multiple receipts.
- Use emojis to make responses friendly.
"""


def build_agent():
    """Build and return a compiled LangGraph ReAct agent."""
    llm = ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0,
    )
    prompt = _build_system_prompt()
    agent = create_react_agent(llm, TOOLS, prompt=prompt)
    return agent


def run_agent(agent, user_message: str, chat_history: list) -> str:
    """Invoke the agent with the user's message and full chat history.
    Returns the agent's final text response.
    """
    # Build the messages list for the agent
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Add the current user message
    messages.append(HumanMessage(content=user_message))

    try:
        result = agent.invoke({"messages": messages})

        # Extract the final AI message from the result
        final_messages = result.get("messages", [])
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return msg.content

        return "I processed your request but couldn't generate a response. Please try again."

    except Exception as e:
        return f"⚠️ Agent error: {e}\n\nMake sure Ollama is running with `llama3.2` model available."

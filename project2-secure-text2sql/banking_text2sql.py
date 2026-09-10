from dotenv import load_dotenv
load_dotenv()
import os
import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate

# ==================== 1. CREATE FAKE BANKING DB ====================
# This is your HSBC-like database

def create_banking_db():
    conn = sqlite3.connect("banking.db")
    cursor = conn.cursor()

    # Drop if exists
    cursor.execute("DROP TABLE IF EXISTS customers")
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP TABLE IF EXISTS accounts")

    # Customers table
    cursor.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT,
        email TEXT,
        phone TEXT,
        risk_rating TEXT,
        branch TEXT
    )
    """)

    # Accounts table
    cursor.execute("""
    CREATE TABLE accounts (
        account_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        account_type TEXT,
        balance REAL,
        currency TEXT,
        status TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
    )
    """)

    # Transactions table - Pillar 3 related
    cursor.execute("""
    CREATE TABLE transactions (
        transaction_id INTEGER PRIMARY KEY,
        account_id INTEGER,
        amount REAL,
        transaction_type TEXT,
        transaction_date TEXT,
        risk_weight REAL,
        capital_charge REAL,
        FOREIGN KEY(account_id) REFERENCES accounts(account_id)
    )
    """)

    # Insert sample data - 10 customers
    customers = [
        (1, 'John Smith', 'john.smith@email.com', '555-0101', 'Low', 'London'),
        (2, 'Sarah Chen', 'sarah.chen@email.com', '555-0102', 'Medium', 'Hong Kong'),
        (3, 'Robert Johnson', 'robert.j@email.com', '555-0103', 'High', 'New York'),
        (4, 'Priya Patel', 'priya.patel@email.com', '555-0104', 'Low', 'Mumbai'),
        (5, 'Ahmed Hassan', 'ahmed.hassan@email.com', '555-0105', 'Medium', 'Dubai'),
    ]

    accounts = [
        (101, 1, 'Savings', 50000, 'GBP', 'Active'),
        (102, 1, 'Current', 25000, 'GBP', 'Active'),
        (103, 2, 'Savings', 120000, 'HKD', 'Active'),
        (104, 3, 'Investment', 500000, 'USD', 'Active'),
        (105, 4, 'Savings', 75000, 'INR', 'Active'),
    ]

    transactions = [
        (1001, 101, 1000, 'Credit', '2026-01-15', 0.2, 200),
        (1002, 101, 500, 'Debit', '2026-01-16', 0.2, 100),
        (1003, 103, 50000, 'Credit', '2026-02-01', 0.5, 25000),
        (1004, 104, 100000, 'Investment', '2026-02-15', 1.0, 100000),
        (1005, 102, 2000, 'Debit', '2026-03-01', 0.2, 400),
    ]

    cursor.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
    cursor.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?)", accounts)
    cursor.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?)", transactions)

    conn.commit()
    conn.close()
    print("Banking DB created with 3 tables: customers, accounts, transactions")

create_banking_db()

# ==================== 2. SECURITY GUARDRAILS ====================
def sql_guardrail_check(question: str, sql_query: str = "") -> tuple[bool, str]:
    # Block prompt injection
    injection_keywords = ["ignore previous", "system prompt", "reveal", "delete data", "drop table", "drop database"]
    if any(k in question.lower() for k in injection_keywords):
        return False, "Blocked: Prompt injection detected"

    # Block destructive SQL
    if sql_query:
        destructive = ["drop", "delete", "truncate", "alter", "update", "insert"]
        if any(k in sql_query.lower() for k in destructive):
            return False, f"Blocked: Destructive operation detected - {sql_query[:50]}"

    return True, "Safe"

def mask_pii_in_result(result: str) -> str:
    # Simple masking for demo - in prod use Presidio again
    import re
    result = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', result)
    result = re.sub(r'\b\d{3}-\d{4}\b', '[PHONE_MASKED]', result)
    return result

# ==================== 3. TEXT-TO-SQL AGENT ====================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

db = SQLDatabase.from_uri("sqlite:///banking.db")

# Custom prompt with banking context
prompt_template = """
You are a secure banking SQL assistant for HSBC.
- Only generate SELECT queries
- Never generate DELETE, DROP, UPDATE, INSERT
- Use JOINs where needed
- For Pillar 3, focus on risk_weight and capital_charge
- If question asks for PII (email, phone), still generate SQL but result will be masked

Database schema:
{custom_table_info}

Question: {input}
"""

agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    agent_type="openai-tools",
    handle_parsing_errors=True
)

# ==================== 4. EVALUATION SET ====================
test_questions = [
    "How many customers are there in total?",
    "What is the total balance across all accounts?",
    "Show me customers with High risk rating",
    "What is the average capital charge for transactions?",
    "Which branch has the most customers?",
    "What is the total risk-weighted assets (sum of capital_charge)?"
]

ground_truth_sql = [
    "SELECT COUNT(*) FROM customers",
    "SELECT SUM(balance) FROM accounts",
    "SELECT * FROM customers WHERE risk_rating = 'High'",
    "SELECT AVG(capital_charge) FROM transactions",
    "SELECT branch, COUNT(*) as cnt FROM customers GROUP BY branch ORDER BY cnt DESC LIMIT 1",
    "SELECT SUM(capital_charge) FROM transactions"
]

print("\n===== TESTING SECURE TEXT-TO-SQL =====")
for i, q in enumerate(test_questions):
    print(f"\n--- Q{i+1}: {q}")

    # Guardrail check
    is_safe, reason = sql_guardrail_check(q)
    if not is_safe:
        print(f"BLOCKED: {reason}")
        continue

    try:
        result = agent.invoke({"input": q})
        masked_result = mask_pii_in_result(result['output'])
        print(f"A: {masked_result[:200]}")
        print(f"Expected SQL: {ground_truth_sql[i]}")
    except Exception as e:
        print(f"Error: {e}")

print("\n===== Project 2 Complete =====")
print("Files created: banking.db, banking_text2sql.py")

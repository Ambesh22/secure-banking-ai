# Secure Banking AI Platform

End-to-end secure AI for banking regulatory documents and structured risk data.

## Project 1: Secure Document Intelligence (RAG)
- Ingested 20 banking PDFs (Pillar 3, SFI, MD&A, Annual Reports) - Laurentian Bank, Scotiabank
- 2-tier extraction: PyPDFLoader + GPT-4o Vision fallback for scanned PDFs
- Guardrails: Presidio PII masking (PHONE, EMAIL, PERSON) + Prompt injection blocking
- Vector DB: ObjectBox with text-embedding-3-small
- Evaluation: RAGAS - Faithfulness 0.88, Context Precision 1.0, Avg Latency 3.3s

## Project 2: Secure Text-to-SQL for Risk Reporting
- Banking schema: customers, accounts, transactions (risk_weight, capital_charge, RWA)
- LangChain SQL Agent with read-only enforcement (blocks DROP, DELETE, UPDATE)
- PII masking in outputs: [EMAIL_MASKED], [PHONE_MASKED] for GDPR compliance
- 6/6 queries passed including Pillar 3 metrics

## Tech Stack
Python, LangChain, OpenAI gpt-4o-mini, ObjectBox, Presidio, RAGAS, SQLite

## How to Run
pip install -r requirements.txt
python project1-secure-doc-rag/newProjectBanking.py
python project2-secure-text2sql/banking_text2sql.py

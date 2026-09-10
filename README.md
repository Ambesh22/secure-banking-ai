# Secure Banking AI Platform

End-to-end secure AI for banking regulatory documents and structured risk data.


## Project 1: Secure Document Intelligence
- 20 Banking PDFs (Pillar 3, SFI, MD&A) - Laurentian Bank, Scotiabank
- Extraction: PyPDFLoader + GPT-4o Vision fallback
- Guardrails: Presidio PII masking + Prompt injection blocking
- VectorDB: ObjectBox, Embeddings: text-embedding-3-small
- **RAGAS Score: Faithfulness 0.88, Context Precision 1.0**

## Project 2: Secure Text-to-SQL
- Schema: customers, accounts, transactions (risk_weight, capital_charge)
- Blocks DROP, DELETE, UPDATE, INSERT (Read-only)
- PII masking: [EMAIL_MASKED], [PHONE_MASKED]
- **6/6 queries passed**

## How to Run
```bash
pip install langchain openai langchain-openai langchain-community presidio-analyzer ragas datasets spacy objectbox
python -m spacy download en_core_web_lg
python secure-banking-ai/project1-secure-doc-rag/newProjectBanking.py
python secure-banking-ai/project2-secure-text2sql/banking_text2sql.py
python project1-secure-doc-rag/newProjectBanking.py
python project2-secure-text2sql/banking_text2sql.py

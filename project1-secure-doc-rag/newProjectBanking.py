from dotenv import load_dotenv
load_dotenv()
import os
import io
import base64
import glob
import tempfile
import atexit
import pandas as pd
from pdf2image import convert_from_path

from typing import Optional, Literal
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.documents import Document

# ==================== BANKING GUARDRAILS ====================
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def mask_pii(text: str) -> str:
    try:
        results = analyzer.analyze(text=text, language='en', entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"])
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except:
        return text

def guardrail_check(question: str) -> tuple[bool, str]:
    injection_keywords = ["ignore previous instructions", "system prompt", "reveal your prompt", "delete data", "drop table"]
    if any(k in question.lower() for k in injection_keywords):
        return False, "Blocked: Prompt Injection detected"
    return True, "Safe"

# ==================== CONFIG ====================
api_key = os.getenv("OPENAI_API_KEY")

class ReportMetadata(BaseModel):
    doc_type: Literal[
        "Annual Report","Quarterly Report","Earnings Press Release","Presentation",
        "Pillar 3 Report","Factbook","MD&A","SFI","SRCI","Other"
    ]
    company_name: Optional[str] = Field(default=None)
    period_ended: Optional[str] = Field(default=None)
    language: str = Field(default="English")

parser = JsonOutputParser(pydantic_object=ReportMetadata)

prompt = ChatPromptTemplate.from_template("""
You are a financial document metadata extractor. Extract ONLY from text. Never guess.
DEFINITIONS:
- Annual Report: Contains "Form 10-K", "Annual Report", "Year Ended"
- Quarterly Report: Contains "Form 10-Q", "Quarterly Report", "Three Months Ended"
- Pillar 3 Report: Contains "Pillar 3", "Basel III"

RULES:
1. Priority: PILLAR3 > SRCI > SFI > FACTBOOK > MDA > Presentation > Earnings Press Release > Quarterly Report > Annual Report
2. If filename contains "Pillar 3" → doc_type="Pillar 3 Report"
3. company_name: Must appear in text

Return STRICT JSON only:
{format_instructions}
Filename: {filename}
Text from first 3 pages:
{document}
JSON:
""").partial(format_instructions=parser.get_format_instructions())

TMP_FILES = []
def cleanup_tmp():
    for f in TMP_FILES:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
atexit.register(cleanup_tmp)

def clean_pdf_header(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read()
        pdf_start = content.find(b'%PDF-')
        if pdf_start in [-1, 0]: return pdf_path
        fd, cleaned_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        TMP_FILES.append(cleaned_path)
        with open(cleaned_path, 'wb') as f:
            f.write(content[pdf_start:])
        return cleaned_path
    except Exception as e:
        return pdf_path

def extract_with_vision(pdf_path):
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=3, dpi=200, fmt="png", thread_count=1)
    except Exception as e:
        print(f"pdf2image failed {os.path.basename(pdf_path)}: {e}")
        return ""
    if not images: return ""
    all_text = []
    vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    for img in images[:3]:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        msg = vision_llm.invoke([{"role": "user", "content": [
            {"type": "text", "text": "Extract all text from this financial document page. Keep layout. Return raw text only."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]}])
        all_text.append(msg.content)
    return "\n".join(all_text)

def extract_text_from_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    if os.path.getsize(pdf_path) > 50 * 1024 * 1024:
        print(f"SKIP: {filename} - too large")
        return ""
    working_path = clean_pdf_header(pdf_path)
    try:
        loader = PyPDFLoader(working_path)
        pages = loader.load()
        text = "\n".join([p.page_content for p in pages])
        if len(text.strip()) > 50:
            print(f"Text PDF: {filename}")
            return text
    except Exception as e:
        print(f"PyPDFLoader failed {filename}: {e}")

    print(f"GPT-4o Vision OCR: {filename}")
    try:
        text = extract_with_vision(working_path)
        return text
    except Exception as e:
        print(f"Vision OCR failed {filename}: {e}")
        return ""

# ==================== MAIN PIPELINE ====================
pdf_paths = glob.glob("research_pdfs/*.pdf")
docs = []
for path in pdf_paths:
    full_text = extract_text_from_pdf(path)
    if full_text:
        masked_text = mask_pii(full_text)
        docs.append(Document(page_content=masked_text, metadata={"source": path, "filename": os.path.basename(path)}))
    else:
        print(f"FAILED: {os.path.basename(path)}")

print(f"Loaded {len(docs)} PDFs with PII masking")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | parser

metadata_results = []
for doc in docs:
    try:
        filename = os.path.basename(doc.metadata["source"])
        is_safe, reason = guardrail_check(filename + " " + doc.page_content[:500])
        if not is_safe:
            print(f"Guardrail Blocked {filename}: {reason}")
            continue
        text = doc.page_content[:18000]
        metadata = chain.invoke({"document": text, "filename": filename})
        metadata['source_file'] = doc.metadata["source"]
        metadata_results.append(metadata)
        print(f"Extracted: {filename} -> {metadata['doc_type']}")
    except Exception as e:
        print(f"Failed {doc.metadata['source']}: {e}")

df = pd.DataFrame(metadata_results)
df.to_csv("report_metadata.csv", index=False)
print(df)

# Vectorstore
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = ObjectBox.from_documents(chunks, embeddings, clear_db=True, embedding_dimensions=1536)
print("Vectorstore created")

# ==================== RAGAS EVALUATION - FIXED ====================
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain.chains import RetrievalQA
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
import time

# Use CONTENT questions only - not filename questions
test_questions = [
    "What is the Pillar 3 disclosure about?",
    "What bank is mentioned in the Pillar 3 report?",
    "What are the key risk metrics disclosed?"
]
ground_truths = [
    "Pillar 3 and Supplementary Regulatory Capital Disclosure",
    "Laurentian Bank of Canada",
    "Key Metrics, Composition of Capital, Leverage Ratio, Credit Risk"
]

qa_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=qa_llm, retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True
)

answers = []
contexts = []
latencies = []

for q in test_questions:
    start = time.time()
    result = qa_chain.invoke({"query": q})
    latency = time.time() - start
    answers.append(result["result"])
    contexts.append([d.page_content for d in result["source_documents"]])
    latencies.append(latency)
    print(f"Q: {q} -> A: {result['result'][:150]} | {latency:.2f}s")

dataset = Dataset.from_dict({
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# FIX for nan
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision], llm=ragas_llm, embeddings=ragas_embeddings)

print("\n===== RAGAS SCORES (For Resume) =====")
print(scores)
scores.to_pandas().to_csv("ragas_scores.csv")
print(f"Avg Latency: {sum(latencies)/len(latencies):.2f}s")

from pypdf import PdfReader
from transformers import pipeline

# load the PDF file
reader = PdfReader("path to the PDF File")

# extract texts from all pages
document_texts = ""
for page in reader.pages:
    document_texts += page.extract_text()


# load the question-answering pipeline
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad"
)

question = ""

# get the answer from Q&A pipeline
result = qa_pipeline(question=question, context=document_texts)
print(f"Answer: {result['answer']}")

from fastapi import FastAPI, UploadFile, Form
from typing import List

app = FastAPI()

@app.post("/answer-questions/")
async def answer_questions(pdf: UploadFile, questions: List[str]):
    
    pdf_path = f"/tmp/{pdf.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())
    
    results = process_pdf_and_questions(pdf_path, questions)
    return results

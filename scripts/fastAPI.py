# from fastapi import FastAPI, UploadFile, Form
# from typing import List
# from main import AgentRAG

# app = FastAPI()
# agent = AgentRAG()

# @app.post("/get_answer/")
# async def answer_questions(pdf: UploadFile, questions: List[str]):
    
#     pdf_path = os.path.join(agent.pdf_root_path,pdf.filename)
#     with open(pdf_path, "wb") as f:
#         f.write(await pdf.read())
    
#     agent.process_pdf(pdf_path)
#     results = agent.get_answer(questions)
#     return results

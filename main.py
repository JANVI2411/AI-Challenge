import os
import time 
import openai
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from pydantic import BaseModel, Field

from scripts.config import Config
from scripts.document import DocumentProcessor
from scripts.vectorstore import VectorStore

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    score: float = Field(
        ...,
        description="A score between 0 and 1 indicating how well the answer resolves the question.",
        ge=0, 
        le=1  
    )

class PDFQuestionAnsweringAgent:

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.pdf_root_path = os.path.join(self.project_dir,"uploaded_pdf")
        if not os.path.exists(self.pdf_root_path):
            os.makedirs(self.pdf_root_path)
        
        self.doc_parser = DocumentProcessor()
        self.vectorstore = VectorStore()
        self.llm = ChatOpenAI(model=Config.LLM_MODEL)
        self.init_filter_doc_chain()
        self.init_answer_grade_chain()
        self.init_rag_llm_chain()

        self.init_agent()

    def init_rag_llm_chain(self):
        system_rag_msg = """You are an intelligent assistant. 
                            Use the provided context to answer the question. 
                            Keep your answers concise, relevant, and accurate."""
        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_rag_msg),
                ("human","Context: {context} \n\n Question:{question} \n\n Answer:")
            ]
        )

        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

    def init_filter_doc_chain(self):
        
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        self.filter_prompt = ChatPromptTemplate.from_messages(
                                [
                                    ("system", system),
                                    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
                                ]
                            )
        self.filter_document_chain = self.filter_prompt | self.llm.with_structured_output(GradeDocuments)
        
    def init_answer_grade_chain(self):
        
        system = """You are a grader evaluating if an answer fully resolves a user's question. 
                    Give a score between 0 to 1:
                    - 1 means the answer completely and accurately resolves the question.
                    - 0 means the answer does not address the question or is incorrect.
                    Rate based on correctness, completeness, and clarity."""

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        
        self.answer_grade_chain = self.answer_prompt | self.llm.with_structured_output(GradeAnswer)
        
    def process_pdf(self,pdf_path):
        chunks = self.doc_parser.process_pdf(pdf_path)        
        coll_name = self.vectorstore.add_docs(chunks)
        return coll_name
    
    def data_retriever(self, question: str):
        documents = self.vectorstore.retrieve_docs(question)
        # print(documents)
        filtered=[]
        for doc in documents:
                grade = self.filter_document_chain.invoke({"document":doc ,"question":question})
                if grade.binary_score=="yes":
                    filtered.append(doc)
        return filtered
    
    def data_retriever_tool(self, question: str):
        '''Tool to retrieve data from the vectore store"'''
        documents = self.vectorstore.retrieve_docs(question)
        return documents

    def init_agent(self):
        tools = [self.data_retriever_tool]
        self.agent_executor = create_react_agent(self.llm, tools)

    def get_answer_react_agent(self,question_list):
        ans_dict = {}
        
        for question in question_list:
            ans = self.agent_executor.invoke({"messages":question})
            generation = ans["messages"][-1].content
            if generation=="":
                ans_dict[question] = "Data Not Available"
                continue 

            grade = self.answer_grade_chain.invoke({"question":question,"generation":generation})
            if grade.score > 0.6:
                ans_dict[question] = generation
            else:
                ans_dict[question] = "Data Not Available"
        return ans_dict 

    def get_answer(self,question_list):
        ans_dict = {} 
        for question in question_list:
            documents = self.data_retriever(question)
            if not documents:
                ans_dict[question] = "Data Not Available"
                continue
            generation = self.rag_chain.invoke({"question":question,"context":"\n".join(documents)})
            grade = self.answer_grade_chain.invoke({"question":question,"generation":generation})
            if grade.score > 0.6:
                ans_dict[question] = generation
            else:
                ans_dict[question] = "Data Not Available"
            
        return ans_dict 

if __name__ == "__main__":
    agent = PDFQuestionAnsweringAgent()
    
    # pdf_path = "uploaded_pdf/handbook.pdf"
    # agent.process_pdf(pdf_path)

    questions = ["What is the name of the company in this document?",
                "Who is the current CEO of the company?",
                "What is their vacation policy?",
                "What is the termination policy?",
                "Who is the president of USA?"]
    ans = agent.get_answer(questions)
    print(ans)
    
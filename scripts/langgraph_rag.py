import os 
import time 
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
import logging 

import json 
from typing import Annotated
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from pymongo import MongoClient

logger = logging.getLogger("rag_logs")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("backend.log")  # Log to a file
handler.setLevel(logging.INFO)

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# os.environ['OPENAI_API_KEY'] = "Your OpenAI API Key"

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

from .config import Config
from .document import DocumentProcessor
from .vectorstore import VectorStore

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

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
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
        self.init_contextualize_chain()
        self.init_rag_llm_chain()
        self.init_rewrite_question_chain()

    def init_rag_llm_chain(self):
        system_rag_msg = """You are an intelligent assistant. 
                            Use the provided context to answer the question. 
                            Provide the answer in a complete sentence.
                            Ensure the answer is concise, relevant, and accurate."""
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
        
    def init_contextualize_chain(self):
        # Prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        self.contextualize_q_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()

    def init_hallucination_chain(self):
    
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
                Give a binary score 'yes' or 'no'. 
                'Yes' means that the answer is grounded in / supported by the set of facts."""
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        self.hallucination_grader_chain = self.hallucination_prompt | self.structured_llm_hallucination_grader
        

    def init_rewrite_question_chain(self):
        
        system_rewrite_question_prompt = """ You are a question re-writer that converts the input question to 
                                            a better version that is optimized for web-search. 
                                            Look at the input and try to reason about the underlying semantic intent meaning."""

        self.rewrite_question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_rewrite_question_prompt),
                ("human", "Initial Question: {question} \n\n Improved Question: "),
            ]
        )
        
        self.rewrite_question_chain = self.rewrite_question_prompt | self.llm | StrOutputParser()

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
    
    def data_retriever_tool(self, question: str):
        '''Tool to retrieve data from the vectore store"'''
        documents = self.vectorstore.retrieve_docs(question)
        return documents


class State(TypedDict):
    # Sequence[BaseMessage]
    questions: Annotated[list[AnyMessage], add_messages]
    docs: Annotated[list[AnyMessage], add_messages]
    filtered_docs: Annotated[list[AnyMessage], add_messages]
    generation: Annotated[list[AnyMessage], add_messages]
    chat_history: list[str]

class ChatBotModelGraph(PDFQuestionAnsweringAgent):
    def __init__(self):
        super().__init__()
        self.workflow = StateGraph(State)
        self.filter_loop_count = 0
        self.hallucination_loop_count = 0
        self.build_graph()

    def contextualize_question(self,state):
        print("---Contextualize Question---")
        query = state["questions"][-1].content  # Using questions instead of messages
        print("----c_query---:",query)
        chat_history = state["chat_history"]
        response = self.contextualize_q_chain.invoke({"chat_history": chat_history, "question": query})
        state["questions"] = [AIMessage(content=response)]
        return state

    def document_retriever(self, state):
        print("---Docs Retriever---")
        
        query = state["questions"][-1].content  # Using questions instead of messages
        docs = self.data_retriever_tool(query)
        state["docs"] = docs  # Store retrieved documents in `docs`
        
        return state

    def filter_documents(self, state):
        print("---FILTER DOCUMENTS---")
        
        query = state["questions"][-1].content
        docs = state["docs"]
        filtered_docs = []
        
        for doc in docs:
            res = self.filter_document_chain.invoke({"document": doc, "question": query})
            score = res.binary_score
            print("########## score:", score)
            if score == "yes":
                filtered_docs.append(doc)
        
        state["filtered_docs"] = filtered_docs  # Store filtered documents in `filtered_docs`
        return state

    def filter_condition_node(self, state):
        print("---FILTER CONDITION NODE---")
        
        if state["filtered_docs"]:
            self.filter_loop_count = 0
            return "rag_generation"
        
        if self.filter_loop_count >= 1:
            self.filter_loop_count = 0
            state["generation"] = [AIMessage(content="Data Not Available")]
            return END
        else:
            self.filter_loop_count += 1
            return "transform_query"

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        
        question = state["questions"][-1].content
        new_question = self.rewrite_question_chain.invoke({"question": question})
        print("------NEW QUERY:", new_question)
        
        state["questions"].append(AIMessage(content=new_question))  # Add transformed query to questions
        return state

    def rag_generation(self, state):
        print("---RAG GENERATION---")
        question = state["questions"][0].content
        docs = "\n".join(state["filtered_docs"][-1].content)
        generation = self.rag_chain.invoke({"context": docs, "question": question})
        print("########", generation)
        
        state["generation"] = [AIMessage(content=generation)]  # Store generated response
        return state

    def hallucination_grader(self, state):
        print("---HALLUCINATION GRADER---")
        
        docs = "\n".join(state["filtered_docs"][-1].content)
        generation = state["generation"][-1].content
        grade = self.hallucination_grader_chain.invoke({"documents": docs, "generation": generation})
        
        if grade.binary_score == "yes":
            self.hallucination_loop_count = 0
            return END
        
        if self.hallucination_loop_count == 1:
            self.hallucination_loop_count = 0
            state["generation"] = [AIMessage(content="[Hallucinated]")]
            return END
        
        self.hallucination_loop_count += 1
        return "rag_generation"

    def build_graph(self):
        self.workflow.add_node("contextualize_question", self.contextualize_question)
        self.workflow.add_node("document_retriever", self.document_retriever)
        self.workflow.add_node("filter_documents", self.filter_documents)
        self.workflow.add_node("transform_query", self.transform_query)
        self.workflow.add_node("rag_generation", self.rag_generation)
        # self.workflow.add_node("hallucination_grader", self.hallucination_grader)
        
        self.workflow.add_edge(START, "contextualize_question")
        self.workflow.add_edge("contextualize_question","document_retriever")
        self.workflow.add_edge("document_retriever", "filter_documents")
        self.workflow.add_conditional_edges("filter_documents", self.filter_condition_node)
        self.workflow.add_edge("transform_query", "document_retriever")
        self.workflow.add_edge("rag_generation", END)
        
        self.app = self.workflow.compile()

class MongoDBHandler:
    def __init__(self, uri="mongodb://localhost:27017", db_name="pdf_chatbot"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def save_session(self, session_id, chat_history):
        """Save session state to MongoDB."""
        self.db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"chat_history": chat_history}},
            upsert=True
        )

    def load_session(self, session_id):
        """Load session state from MongoDB."""
        session = self.db.sessions.find_one({"session_id": session_id})
        return session["chat_history"] if session else []

    def delete_session(self, session_id):
        """Delete a session from MongoDB."""
        self.db.sessions.delete_one({"session_id": session_id})

class ManageHistory:
    def __init__(self):
        self.MAX_CHAT_HISTORY_LENGTH = 4
        self.thresh = 2
        openai_model,temperature = 'gpt-4o-mini', 0
        self.llm = ChatOpenAI(model=openai_model, temperature=temperature)
        
        self.system_prompt = """Given the chats between human and AI, your task is to summarize 
                        their conversation into 50 tokens """
        self.prompt = ChatPromptTemplate.from_messages(
                        [("system",self.system_prompt ),
                        ("user","Chat: {chat}")])
        self.summary_llm = self.prompt | self.llm | StrOutputParser()

    def summarize_chat_history(self,chat):
        summary = self.summary_llm.invoke({"chat":chat})
        return summary

    def manage_chat_history(self,chat_history):
        if len(chat_history) >= self.MAX_CHAT_HISTORY_LENGTH:
            chat_history_summary = self.summarize_chat_history(chat_history[:self.thresh])
            chat_history = [f"Chat Summary: {chat_history_summary}"] + chat_history[self.thresh:]
        return chat_history

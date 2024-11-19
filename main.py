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
from langchain_core.messages import AIMessage, HumanMessage
from scripts.langgraph_rag import ChatBotModelGraph,ManageHistory
import json 

class ChatBot:
    def __init__(self):
        self.chatbot = ChatBotModelGraph()
        # self.manage_history = ManageHistory()

    def askModel(self,user_question):
        question = AIMessage(content=user_question)
        
        # chat_history = self.user_sessions[user_id].get_chat_history()
        # chat_history = self.manage_history.manage_chat_history(chat_history)
        chat_history = []
        response = self.chatbot.app.invoke({"chat_history":chat_history,"questions": [question]}) 

        msg = ""
        if not response["generation"]: 
            msg, status=  "Data Not Available", "failed"
        else:
            msg, status = response["generation"][-1].content, "success"
        
        # chat_history.append(f"Human: {query}")
        # chat_history.append(f"Answer: {msg}")
        
        # self.self.user_sessions[user_id].update_chat_history(chat_history)
        return msg,status
    
    def process_pdf(self,pdf_path):
        self.chatbot.process_pdf(pdf_path)

    def get_answer(self,question_list):
        ans_dict = {} 
        for question in question_list:
            generation, status = self.askModel(question)
            ans_dict[question] = generation
        return ans_dict 

if __name__ == "__main__":
    chatbot = ChatBot()

    pdf_path=""
    chatbot.process_pdf(pdf_path)

    questions = ["What is the name of the company?",
                "Who is the CEO of the company?",
                "What is their vacation policy?",
                "What is the termination policy?",
                "Who is the president of USA?"]
    ans_dict = chatbot.get_answer(questions)
    print(ans_dict)
    """
    Generated Answer:
    {
    'What is the name of the company?': 'The name of the company is Zania, Inc.', 
    
    'Who is the CEO of the company?': 'The CEO of the company is Shruti Gupta.', 
    
    'What is their vacation policy?':'The vacation policy is outlined in section 7.7.', 
    
    'What is the termination policy?': 'The termination policy states that involvement in criminal activity while employed by the company may result in disciplinary action, including suspension or termination of employment.', 
     
     'Who is the president of USA?': 'Data Not Available'
     }
    """

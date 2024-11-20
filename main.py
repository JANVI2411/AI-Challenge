import os
import time 
from langchain_core.messages import AIMessage, HumanMessage
from scripts.langgraph_rag import AgentGraph
# from scripts.langgraph_rag ManageHistory,MongoDBHandler
import json 
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

class AgentRAG:
    def __init__(self):
        self.agent_rag = AgentGraph()
        self.session_id = str(uuid.uuid4())
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        self.pdf_root_path = os.path.join(self.project_dir,"uploaded_pdf")
        if not os.path.exists(self.pdf_root_path):
            os.makedirs(self.pdf_root_path)

        '''
        self.manage_history = ManageHistory()
        self.memory = MongoDBHandler()
        self.chat_history = self.memory.load_session(self.session_id)
        '''

    def askModel(self,user_question):
        question = AIMessage(content=user_question)
        
        '''
        self.chat_history = self.manage_history.manage_chat_history(self.chat_history)
        '''
        self.chat_history = []
        response = self.agent_rag.app.invoke({"chat_history":self.chat_history,"questions": [question]}) 

        msg = ""
        if not response["generation"]: 
            msg, status=  "Data Not Available", "failed"
        else:
            msg, status = response["generation"][-1].content, "success"
        
        '''
        self.chat_history.append(f"Human: {question}")
        self.chat_history.append(f"Answer: {msg}")
        
        self.memory.save_session(self.session_id,self.chat_history)
        '''
        return msg,status
    
    def process_pdf(self,pdf_path):
        self.agent_rag.process_pdf(pdf_path)

    def get_answer(self,question_list):
        ans_dict = {} 
        for question in question_list:
            generation, status = self.askModel(question)
            ans_dict[question] = generation
        return ans_dict 

if __name__ == "__main__":
    agent_rag = AgentRAG()

    pdf_path="uploaded_pdf/handbook.pdf"
    agent_rag.process_pdf(pdf_path)

    questions = ["What is the name of the company?",
                "Who is the CEO of the company?",
                "What is their vacation policy?",
                "What is the termination policy?",
                "Who is the president of USA?"]
    ans_dict = agent_rag.get_answer(questions)
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

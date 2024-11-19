# https://cookbook.openai.com/examples/vector_databases/chroma/using_chroma_for_embeddings_search

import openai
import os
import uuid
import shutil
import chromadb
from .config import Config
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import warnings
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

openai.api_key = os.getenv("OPENAI_API_KEY")
    
class VectorStore:
    def __init__(self,embedding_model = Config.EMBEDDING_MODEL):
        self.chroma_client = chromadb.PersistentClient()
        self.embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=embedding_model)
        self.coll_name = None 
        self.top_k = 5

    def add_docs(self,docs):
        print("\nProcessing PDF...")
        self.coll_name = 'pdf_data_' + str(uuid.uuid4())
        self.content_collection = self.chroma_client.get_or_create_collection(name=self.coll_name, embedding_function=self.embedding_function)

        doc_ids = [f"{uuid.uuid4()}" for i in range(len(docs))]
        self.content_collection.upsert(
            documents=docs,   
            ids=doc_ids    
        )

    def retrieve_docs(self,query):
        self.coll_name = "pdf_data_1e663c93-5ecc-410e-a663-7b026d189869"
        print("Retrieving docs...")
        # if self.coll_name:
        self.content_collection = self.chroma_client.get_or_create_collection(name=self.coll_name, embedding_function=self.embedding_function)
        results = self.content_collection.query(query_texts=query, n_results=self.top_k, include=['documents']) 
        return results["documents"][0]
        # return []
       
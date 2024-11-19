import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=50
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL="text-embedding-3-small" #"text-embedding-3-small"
    LLM_MODEL="gpt-4o-mini"
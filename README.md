# PDF Question-Answering App (LLM + RAG + LangChain + LangGraph + Streamlit + FastAPI)

This project allows users to upload the PDF and the questions via a Streamlit frontend. The LangGraph and LangChain backend processes the uploaded PDF, creates a vector store using ChromaDB, and saves the document. Users can ask questions related to the PDF, and the app provides accurate answers using Retrieval-Augmented Generation (RAG) pipeline.

I have implemented,
1) query transformation, document filtering, and answer relevancy checks to enhance the accuracy of responses.
2) the agent using LangGraph to create the graph flow, ensure accurate decision-making, and streamline the workflow.

## Features

- **PDF Upload**: Upload the PDF file and questions through the frontend (Streamlit).
- **Backend PDF Processing**: The backend will process the uploaded PDF, convert it into embeddings, and store it in a ChromaDB vector store.
- **Question-Answering System**: User can ask questions related to the content of the PDF, and the system will answer based on the processed data using RAG.
- **Language Model**: OpenAI gpt-4o-mini model is used to generate responses based on the retrieved document embeddings.

## Tech Stack

- **Streamlit**
- **FastAPI**
- **LangChain**
- **LangGraph**
- **Vector Store (ChromaDB)**

## Installation

**Prerequisites**: Python 3.8 or later

### Clone the repository

Clone the repository
```bash
git clone https://github.com/JANVI2411/LLM-PDF-QA-Summarizer.git
cd ai-challenge
```

Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit
```

### Usage

Start the Streamlit app

```bash
streamlit run streamlit_app.py
```

This will launch the Streamlit UI where you can upload PDFs and ask questions.


## Future Enhancements
- Advanced search: Improve the vector search with better ranking algorithms.
- Query Batching and Parallelism to reduce answer retrieval time.


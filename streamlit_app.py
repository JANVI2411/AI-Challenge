import streamlit as st
from main import PDFQuestionAnsweringAgent
import os 

agent = PDFQuestionAnsweringAgent()

st.title("PDF Question-Answering Agent")
uploaded_file = st.file_uploader("Upload a PDF")
if "questions" not in st.session_state:
    st.session_state.questions = []

# Step 2: Interactive Question Input
if uploaded_file:
    st.subheader("Ask Questions")
    question = st.text_input("Enter a question:")

    # Button to add the question to the list
    if st.button("Add Question"):
        if question:
            st.session_state.questions.append(question)
            st.success("Question added!")
        else:
            st.error("Please enter a question.")

    # Display the list of added questions
    if st.session_state.questions:
        st.write("Questions:")
        for i, q in enumerate(st.session_state.questions, 1):
            st.write(f"{i}. {q}")

    # Final submission button to process all questions
    if st.button("Submit All Questions"):
        if st.session_state.questions:
            pdf_root_path = agent.pdf_root_path
            pdf_path = os.path.join(pdf_root_path,uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            with st.spinner("Processing your PDF and questions..."):
                agent.process_pdf(pdf_path)
                results = agent.get_answer(st.session_state.questions)
            st.json(results)
            st.session_state.questions=[]
        else:
            st.error("Please add at least one question before submitting.")

# questions = st.text_area("Enter questions (comma-separated):")

# if st.button("Submit"):
#     if uploaded_file and questions:
#         pdf_root_path = agent.pdf_root_path
#         pdf_path = os.path.join(pdf_root_path,uploaded_file.name)
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.read())
        
#         agent.process_pdf(pdf_path)
#         results = agent.get_answer(questions.split(","))
#         st.json(results)

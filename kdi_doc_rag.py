import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI


import os

# Set OpenAI API key - GET THIS FROM SECRETS.PY

# Streamlit UI
st.title("KDI Document AI Agent")
st.write("Upload a Word document and ask questions!")
# Upload Word document
uploaded_file = st.file_uploader("Upload a Word document", type=["docx"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.docx", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the Word document
    loader = Docx2txtLoader("temp.docx")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and store in FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Initialize the retriever
    retriever = vectorstore.as_retriever()

    # Initialize the LLM and QA chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # User input for question
    question = st.text_input("Ask a question about the document:")

    if question:
        # Get the answer from the QA chain
        answer = qa_chain.run(question)
        st.write("**Answer:**")
        st.write(answer)
import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

st.title("📄 Chat with your PDF - Powered by Gemini Flash 1.5")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # Split the text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Take a user query
    query = st.text_input("Ask a question about the PDF")
    if query:
        docs = vectorstore.similarity_search(query)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        st.write("📌 Answer:", response)

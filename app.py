import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



st.title("Chatbot Applikation med Rag")

loader = PyPDFLoader("Foretagsinfo_RSA_AB.pdf")
Pdf_data = loader.load()  # entire PDF is loaded as a single Document

#split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(Pdf_data)

load_dotenv() 
# Get the value
api_key = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vector = embeddings.embed_query("hello, world!")
#vector[:5]

vectorstore = InMemoryVectorStore.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Retrieve the most similar text
retrieved_documents = retriever.invoke("Hur kan jag kontakta er support")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    
)

query = st.chat_input("Du kan fråga något: ") 
prompt = query
# System prompt
system_prompt = (
"Du är en assistent för att besvara frågor."
"Använd följande delar av hämtad kontext för att besvara frågan."
"Om du inte vet svaret, säg att du inte vet."
"Använd högst tre meningar och håll svaret kortfattat.""\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    #print(response["answer"])

    st.write(response["answer"])
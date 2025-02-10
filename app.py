import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv(key='HF_TOKEN')








st.title('Conversational RAG with PDF uploads')
st.write("Upload PDF's and chat with their content")

# Groq API Key
groq_api_key=st.text_input(label='Enter your Groq API Key',type='password')

if groq_api_key: 

    #model
    llm=ChatGroq(model_name='Gemma2-9b-It',groq_api_key=groq_api_key)   

    #session_id
    session_id=st.text_input(label='Session ID',value='default_session')

    #file upload
    uploaded_files=st.file_uploader(label='Choose a PDF file',type='pdf',accept_multiple_files=True)
    
    if uploaded_files:
        documents=list()
        for uploaded_file in uploaded_files:
            temp_dir="./temp.pdf"
            with open(file=temp_dir,mode='wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(file_path=temp_dir)
            docs=loader.load()    
            documents.extend(docs)

        # data--> text chunks
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
        final_doc=text_splitter.split_documents(documents=documents)
        # text --> vectors and store it into Chroma DB
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore=FAISS.from_documents(documents=final_doc,embedding=embedding)
        retriever=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human','{input}')
        ]
    )

        history_aware_retriever=create_history_aware_retriever(llm=llm,retriever=retriever,prompt=contextualize_q_prompt)

        # Question-Answer
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    
        qa_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human','{input}')
        ]
    )
    
        document_chain=create_stuff_documents_chain(llm=llm,prompt=qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,document_chain)

        if 'store' not in st.session_state:
            st.session_state.store=dict()
            
        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
                                      
        conversational_rag_chain=RunnableWithMessageHistory(rag_chain,
                                                        get_session_history,
                                                        input_messages_key='input',
                                                        history_messages_key='chat_history',
                                                        output_messages_key='answer')                                                                       
    
        user_input=st.text_input(label='Your Question:')

        if user_input:
            response=conversational_rag_chain.invoke(
            input={'input':user_input},
            config={'configurable':{'session_id':session_id}}
            )

            session_history=get_session_history(session_id=session_id)

            st.write(st.session_state.store)
            st.write('Assistant:',response.get('answer'))
            st.write('Chat History:',session_history.messages)

else:
    st.warning('Enter a valid GROQ API Key')       
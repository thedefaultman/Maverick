#sys imports
import os
from dotenv import load_dotenv
#non-sys imports LOL
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith import Client

from langchain_pinecone import Pinecone
import pinecone


#initialize your stuff here i.e langsmith, load_dotenv, pinecone
load_dotenv()
client = Client()
api_key = os.environ.get("PINECONE_API_KEY")
pinecone_host = os.environ.get("PINECONE_HOST")
pinecone_env = os.environ.get("PINECONE_ENVIRONMENT_REGION")






#piping the 3 function below toghether to finetune the answer. 
#first get the url and dishout vector_store
#second get vector_store and dishout a retriever_chan and dishout the final answer

def get_vectorstore_from_url(url):
    # get the text in doc form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split doc into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    document_chunks = text_splitter.split_documents(document)
    
    
    #setup the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    #Defining our pinecone index
    index = pinecone.Index(
        api_key=api_key,
        host=pinecone_host,
        index_name="marvin",
        environment= pinecone_env
    )
    
    #now we create a pinecone object and pass it the index and embedding that we created above
    pc = Pinecone(index=index, embedding=embeddings)
    vector_store = pc.from_documents(documents=document_chunks, embedding=embeddings, index_name="marvin")
    return vector_store
    
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    #if the variable chat_history exists, prompt will be populated with it's value
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    return create_retrieval_chain(retriever=retriever_chain, combine_docs_chain=stuff_document_chain)
    
def get_response(user_input):
    #create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vectore_store)
    #talk to Patrick on how to move this to the backend and expose just the invoke method
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']


# app config
st.set_page_config(page_title="MAVERICK", page_icon="🤖")
st.title("MAVERICK: Chat with your domain")


# sidebar
with st.sidebar:
    st.header("setting")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
    
else:
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello I'm a Maverick. How can I help you?"),
        ]
    
    # This will make the document to presist. It's not gonna spam the pinecone steps
    if "vectore_store" not in st.session_state:
        st.session_state.vectore_store = get_vectorstore_from_url(website_url)
        


    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # setting the response 
        response = get_response(user_query)
        # Appending the user input to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # Appending the AI response to the chat history
        st.session_state.chat_history.append(AIMessage(content=response))
        
        

    # Conversation

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

import os
from PIL import Image
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from supabase import Client, create_client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory


load_dotenv()

documents = []

# Connect to Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)



def choose_llm_model(llm:str, temperature:float):
    """
    Chooses the LLM model based on the input parameter.

    Args:
        llm (str): The name of the LLM model.
        temperature (float): The temperature of the LLM model.

    Returns:
        An instance of the selected LLM model and its temperature.
    """
    if llm == 'ChatGPT-4':
        return ChatOpenAI(model="gpt-4-0125-preview", temperature=temperature)
    
    if llm == 'ChatGPT-3.5':
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    
    elif llm == 'Gemini-Pro':
        if temperature >= 1:
            temperature = 1
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature, convert_system_message_to_human=True)
    
    elif llm == 'Llama2':
        return ChatGroq(model_name="llama2-70b-4096", temperature=temperature)
    
    elif llm == 'Mistral':
        return ChatGroq(model_name="mixtral-8x7b-32768", temperature=temperature)
    
    elif llm == 'Gemma':
        return ChatGroq(model_name="gemma-7b-it", temperature=temperature)
    
    
    

def load_vectorstore_database():
    
    vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    table_name="documents",
    query_name="match_documents",
)
    
    return vector_store


def get_vectorstore_from_pdf_files(pdf_files):
    text = ''
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
                text += page.extract_text() + '\n\n'
                
        # split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=128)
        document_chunks = text_splitter.split_text(text)
        
        # create a vectorstore from the chunks
        vector_store = SupabaseVectorStore.from_texts(
            document_chunks,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
            )

    return vector_store



def get_vectorstore_from_folder_url(folder_url):
    # get the text in document form
    for file in os.listdir(folder_url):
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(folder_url, file)
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                    
                    # split the document into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=128)
                    document_chunks = text_splitter.split_documents(documents)
                    
                    # create a vectorstore from the chunks
                    vector_store = SupabaseVectorStore.from_documents(
                        document_chunks,
                        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
                        client=supabase,
                        table_name="documents",
                        query_name="match_documents",
                        chunk_size=500,
                        )

    return vector_store

def get_vectorstore_from_website_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=128)
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = SupabaseVectorStore.from_documents(
        document_chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=500,
        )

    return vector_store

def get_context_retriever_chain(vector_store, model, temperature):
    
    llm = choose_llm_model(model, temperature)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation if you cannot found a relevant ifnormation then return information not found."),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ])
    
    
    history_aware_retriever  = create_history_aware_retriever(llm, retriever, prompt)
    
    return history_aware_retriever 
    
    
    
def get_conversational_rag_chain(history_aware_retriever, model, temperature): 
    
    llm = choose_llm_model(model, temperature)
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context if there is no context or if you do not know the answer tell the user you do not know, do not create any informatin out of the context:\n\n{context}"),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ])
    
    question_answer_chain  = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def conversation_rag_chain(rag_chain):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )
    
    return conversational_rag_chain



def get_response(user_input, vectore_store, llm, temperature):
    
    # try:
    # retriever_chain = get_context_retriever_chain(vectore_store, llm, temperature)
    # rag_chain = get_conversational_rag_chain(retriever_chain, llm, temperature)
    # conversational_rag_chain = conversation_rag_chain(rag_chain)
    
    # except:
        
    try:
        # Attempt to retrieve the context retriever chain
        retriever_chain = get_context_retriever_chain(vectore_store, llm, temperature)
        
        # Attempt to retrieve the conversational RAG chain
        rag_chain = get_conversational_rag_chain(retriever_chain, llm, temperature)
        
        # Attempt to create the conversational RAG chain
        conversational_rag_chain = conversation_rag_chain(rag_chain)
    
        response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": "abc123"}
        },)["answer"]
        
        return response
    
    except Exception as e:
        # Handle any exceptions that occur during the process
        return "ChatGPT-4 is on the way, consider using a different model. If errors persist with another model, reduce the temperature setting."
    
    
    
    
    
    
def chat_bot(pdf_url, pdf_files, website_url, open_chat, model, temperature):
    with st.spinner("Processing..."):
            
            # session state
            if open_chat:
                if "vector_store" not in st.session_state:
                    st.session_state['vector_store'] = load_vectorstore_database()
                    
            if pdf_files:
                if "vector_store" not in st.session_state:
                    st.session_state['vector_store'] = get_vectorstore_from_pdf_files(pdf_files)    
            
            if pdf_url:
                if "vector_store" not in st.session_state:
                    st.session_state['vector_store'] = get_vectorstore_from_folder_url(pdf_url)    
                
            elif website_url:
                if "vector_store" not in st.session_state:
                    st.session_state['vector_store'] = get_vectorstore_from_website_url(website_url)    
                
            if "chat_history" not in st.session_state:
                st.session_state['chat_history'] = [
                    AIMessage(content="Hello, I am a bot. How can I help you?"),
                ]
                


            # user input
            user_query = st.chat_input("Type your message here...")
            if user_query is not None and user_query != "":
                response = get_response(user_query, st.session_state['vector_store'], model, temperature)
                st.session_state.chat_history.append(HumanMessage(user_query))
                st.session_state.chat_history.append(AIMessage(response))
                        
                        
            # Conversation
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        if message.content == 'ChatGPT-4 is on the way, consider using a different model. If errors persist with another model, reduce the temperature setting.':
                            st.error(message.content)
                        else:
                            st.markdown(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.markdown(message.content)



def main():

    # app config
    im = Image.open("icon32.ico")
    st.set_page_config(page_title="Chat with websites", page_icon=im)
    st.title("Chat Bot")



    with st.sidebar:
        st.header("Settings")
        if open_chat := st.toggle('Use the Existing Database'):
            st.success('Successfully Connected!')
        url_type = st.selectbox('Choose URL type', ('PDF_url', 'PDF_files', 'Website_url'))
        
        
        if url_type == 'PDF_url':
            pdf_files = []
            website_url = ''
            pdf_url = st.text_input("PDF Folder URL").replace('\\','/')
        
        elif url_type == 'Website_url':
            pdf_url = ''
            pdf_files = []
            website_url = st.text_input("Website URL")
        
        elif url_type == 'PDF_files':
            pdf_url = ''
            website_url = ''
            pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
            
        st.button('Submit')
        st.divider()
        
        st.subheader("Select AI Model & Temperature")
        model = st.selectbox('Default: OPENAI ChatGPT 3.5', ('ChatGPT-3.5', 'ChatGPT-4', 'Gemini-Pro',  'Llama2', 'Mistral', 'Gemma'))
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        
        
    if not open_chat:
        if url_type == 'PDF_url' and (pdf_url is None or pdf_url == '' or '/' not in pdf_url):
            st.info("Please Enter a Valid Folder URL")
            
        elif url_type == 'Website_url' and (website_url is None or website_url == '' or 'https://' not in website_url):
            st.info("Please Enter a Valid Website URL")

        elif url_type == 'PDF_files' and (pdf_files is None or pdf_files == []):
            st.info("Please Upload PDF files")

        else:
            chat_bot(pdf_url, pdf_files, website_url, open_chat, model, temperature)
            
            
    else:
        

        if url_type == 'PDF_url' and pdf_url and '/' not in pdf_url:
            st.info("Please Enter a Valid Folder URL")
            
        elif url_type == 'Website_url' and website_url and 'https://' not in website_url:
            st.info("Please Enter a Valid Website URL")

        elif url_type == 'PDF_files' and not pdf_files:
            st.info("Please Upload PDF files")

        else:
            chat_bot(pdf_url, pdf_files, website_url, open_chat, model, temperature)
    
if __name__ == '__main__':
    main()
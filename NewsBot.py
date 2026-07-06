import os
import time
import requests
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

MAX_LIMIT = 25
load_dotenv()

st.title("News Research Assistant")
st.caption("Ask questions based on provided news articles. The database resets when you refresh!")


""" Check if vectorestore is present, if not, initialize it to None. 
    This is important because the vectorstore is used to store the embeddings of the 
    processed articles, and we want to ensure that it is initialised so that articles
    are stored seamlessly."""


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please process some URLs in the sidebar, then ask me anything about them."}]

if "article_count" not in st.session_state:
    st.session_state.article_count = 0

# Cache the embedding model globally so it only loads into memory once to improve performance and reduce latency.
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embedding_model()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
    google_api_key=st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)


# UI code:

st.sidebar.title("Enter News URLs 🔍")
st.sidebar.markdown(f"**Total Articles in Database:** `{st.session_state.article_count}/{MAX_LIMIT}`")

with st.sidebar.form(key="url_form", clear_on_submit=True):
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2")
    url3 = st.text_input("URL 3")
    process_url_clicked = st.form_submit_button("Process URLs", icon="📥")

st.sidebar.markdown("---")
reset_db_clicked = st.sidebar.button("Reset My Database", icon="🗑️")

def get_clean_url(url):
    if "google.com/url" in url:
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            if "url" in query_params:
                return query_params["url"][0]
        except:
            return url
    return url

if reset_db_clicked:
    st.session_state.vectorstore = None
    st.session_state.article_count = 0
    st.session_state.messages = [{"role": "assistant", "content": "Database cleared! Ready for new URLs."}]
    st.rerun()

# Logic for processing articles to be stored in vectorstore and then used for RAG.
# For seamless text extraction, Jina AI reader API used as python loaders cause issues and are not reliable for all websites. 

if process_url_clicked:
    if st.session_state.article_count >= MAX_LIMIT:
        st.sidebar.warning(f"You've reached the maximum limit of {MAX_LIMIT} articles. Please reset your database to add more.", icon="⚠️")
    else:
        raw_urls = [url1, url2, url3]
        urls = [url.strip() for url in raw_urls if url.strip()]
    
        if not urls:
            st.sidebar.warning("Please enter at least one URL.", icon="⚠️")
        else:
            all_docs = []
            processed_count = 0
            
            with st.spinner("Fetching, chunking, and embedding articles via Jina API..."):
                for url in urls:
                    clean_url = get_clean_url(url)
                    jina_api_url = f"https://r.jina.ai/{clean_url}"
                    
                    try:
                        response = requests.get(jina_api_url, timeout=15)
                        if response.status_code == 200 and response.text.strip():
                            doc = Document(page_content=response.text, metadata={"source": clean_url})
                            all_docs.append(doc)
                            processed_count += 1
                        else:
                            st.sidebar.error(f"Failed to fetch content from {urlparse(clean_url).netloc} (Status: {response.status_code})", icon="🚫")
                    except Exception as e:
                        st.sidebar.error(f"Error reaching Jina API for {urlparse(clean_url).netloc}: {e}", icon="⚠️")

                if not all_docs:
                    st.sidebar.error("No valid data retrieved.", icon="❌")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
                    docs = text_splitter.split_documents(all_docs)

                    if docs:
                        if st.session_state.vectorstore is None:
                            st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                        else:
                            st.session_state.vectorstore.add_documents(docs)
                        
                        st.session_state.article_count += processed_count
                        
                        alert_container = st.sidebar.empty()
                        alert_container.success(f"{processed_count} article(s) added!", icon="✅")
                        time.sleep(2.5)
                        alert_container.empty()
                        st.rerun()
                    else:
                        st.sidebar.error("Failed to split documents.", icon="❌")


# Chat interface for user queries and RAG response generation.

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your articles..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore is None or st.session_state.article_count == 0:
        error_msg = "⚠️ Your database is empty. Please process some URLs in the sidebar first!"
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system_prompt = (
                    "You are a helpful assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If the answer is not in the context, strictly say 'Info regarding this topic not present'. "
                    "DO NOT make up an answer."
                    "\n\n"
                    "{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                rag_chain = RunnableParallel(
                    {"context": retriever, "input": RunnablePassthrough()}
                ).assign(answer=rag_chain_from_docs)

                try:
                    response = rag_chain.invoke(query)
                    answer_text = response["answer"]
                    
                    if "Info regarding this topic not present" not in answer_text:
                        sources = set()
                        for doc in response.get("context", []):
                            if "source" in doc.metadata:
                                sources.add(doc.metadata["source"])
                        
                        if sources:
                            answer_text += "\n\n**Sources:**\n"
                            for source in sources:
                                domain = urlparse(source).netloc
                                answer_text += f"* [{domain}]({source})\n"

                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        st.error("Free rate limit reached. Please wait a minute before sending your next request.", icon="⏳")
                    else:
                        st.error(f"An error occurred: {e}")
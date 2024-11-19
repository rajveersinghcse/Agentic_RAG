import streamlit as st
import os
import PyPDF2
from tqdm import tqdm
import re
import json
import requests
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import completion
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None


def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    urls.add(
                        parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    )
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())

            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        else:
            st.warning(
                f"Failed to fetch content from {url}: Status code {response.status_code}"
            )
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None


def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None


def get_embeddings(texts, model="text-embedding-3-small", api_key=None):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"input": texts, "model": model}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["data"]
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None


def process_uploaded_pdfs(uploaded_files):
    pdf_list = []
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text()
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return pdf_list


def process_and_index_documents(
    uploaded_files, web_urls=None, chunk_size=150, crawl_website=False
):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    all_chunks = []
    doc_metadata = []

    if uploaded_files:
        all_documents = process_uploaded_pdfs(uploaded_files)
        for doc in all_documents:
            chunks = text_splitter.split_text(doc["content"])
            all_chunks.extend(chunks)
            for _ in chunks:
                doc_metadata.append(
                    {"filename": doc["filename"], "source": "pdf_dataset"}
                )

    if web_urls:
        urls = [url.strip() for url in web_urls.split(",")]

        if crawl_website:
            all_urls = set()
            progress_bar = st.progress(0)
            progress_text = st.empty()

            for i, base_url in enumerate(urls):
                progress_text.text(f"Crawling website: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                progress_bar.progress((i + 1) / len(urls))

            urls = list(all_urls)
            progress_text.text(f"Found {len(urls)} unique URLs")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, url in enumerate(urls):
            progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)

            if content is not None:
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)
                for _ in chunks:
                    doc_metadata.append({"url": url, "source": "web_content"})

            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)

        progress_text.empty()
        progress_bar.empty()

    if not all_chunks:
        st.error("No content to process. Please provide valid PDFs or web URLs.")
        return None, None

    api_key = st.session_state.openai_api_key

    with st.spinner("Generating embeddings..."):
        embeddings_objects = get_embeddings(all_chunks, api_key=api_key)
        if not embeddings_objects:
            return None, None
        embeddings = [obj["embedding"] for obj in embeddings_objects]

    client = QdrantClient("http://localhost:6333")
    collection_name = "agent_rag_index"
    VECTOR_SIZE = 1536

    with st.spinner("Creating vector database..."):
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [
            {"content": chunk, "metadata": metadata}
            for chunk, metadata in zip(all_chunks, doc_metadata)
        ]

        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )

    st.success(
        f"Indexed {len(all_chunks)} chunks from {len(set(m['source'] for m in doc_metadata))} different sources"
    )
    return client, collection_name


def answer_question(question, client, collection_name, top_k=3):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    def search(text: str):
        query_embedding = get_embeddings(text, api_key=st.session_state.openai_api_key)[
            0
        ]["embedding"]
        return client.search(
            collection_name=collection_name, query_vector=query_embedding, limit=top_k
        )

    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            source_info = ""
            if doc.payload["metadata"]["source"] == "pdf_dataset":
                source_info = (
                    f"\nSource: PDF file {doc.payload['metadata']['filename']}"
                )
            else:
                source_info = f"\nSource: Web article {doc.payload['metadata']['url']}"
            formatted_chunks.append(doc.payload["content"] + source_info)
        return "\n\n".join(formatted_chunks)

    decision_system_prompt = """Your job is decide if a given question can be answered with a given context. 
    If context can answer the question return 1.
    If not return 0.
    Context: {context}
    """

    system_prompt = """You are an expert in answering questions. Provide answers based **exclusively** on the given context. 

        **Rules:**
        1. If the question cannot be answered using the context, respond only with: "I don't know."
        2. Do **not** infer, assume, or add information not explicitly provided in the context.
        3. Your answers must be:
        - **Concise**: Avoid unnecessary details.
        - **Informative**: Focus on actionable and precise responses.
        4. Format your response in **Markdown**.

        **Context:** {context}

    """

    user_prompt = """
    Question: {question}
    Answer:"""

    with st.spinner("Searching for relevant information..."):
        results = search(question)
        context = format_docs(results)

        response = completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "content": decision_system_prompt.format(context=context),
                    "role": "system",
                },
                {"content": user_prompt.format(question=question), "role": "user"},
            ],
            max_tokens=50,
            api_key=st.session_state.openai_api_key,
        )
        has_answer = response.choices[0].message.content

        if has_answer == "1":
            st.info("Found relevant information in the indexed content")
            response = completion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "content": system_prompt.format(context=context),
                        "role": "system",
                    },
                    {"content": user_prompt.format(question=question), "role": "user"},
                ],
                max_tokens=1000,
                api_key=st.session_state.openai_api_key,
            )
            st.markdown(response.choices[0].message.content)
        else:
            st.info("No relevant information found. Searching online...")
            results = DDGS().text(question, max_results=5)
            context = "\n\n".join(doc["body"] for doc in results)
            st.info("Found online sources. Generating the response...")
            response = completion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "content": system_prompt.format(context=context),
                        "role": "system",
                    },
                    {"content": user_prompt.format(question=question), "role": "user"},
                ],
                max_tokens=1000,
                api_key=st.session_state.openai_api_key,
            )
            st.markdown(response.choices[0].message.content)


st.title("RAG System with PDF and Website Crawling Support")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    st.session_state.openai_api_key = api_key

uploaded_files = st.file_uploader(
    "Upload PDF files:", accept_multiple_files=True, type=["pdf"]
)

st.subheader("Website Input")
web_urls = st.text_input(
    "Enter website URLs (comma-separated):", placeholder="https://example.com"
)
crawl_website = st.checkbox(
    "Crawl entire website(s)",
    help="Enable this to extract content from all pages of the specified website(s)",
)

if st.button("Process and Index Documents"):
    if not st.session_state.get("openai_api_key"):
        st.error("Please enter your OpenAI API key first.")
    else:
        st.session_state.client, st.session_state.collection_name = (
            process_and_index_documents(
                uploaded_files, web_urls, crawl_website=crawl_website
            )
        )

if st.session_state.client and st.session_state.collection_name:
    question = st.text_input("Ask a question about the documents:")
    if st.button("Get Answer"):
        answer_question(
            question, st.session_state.client, st.session_state.collection_name
        )
elif uploaded_files or web_urls:
    st.warning("Please process and index the documents first.")
else:
    st.info("Upload PDFs or provide web URLs to get started.")

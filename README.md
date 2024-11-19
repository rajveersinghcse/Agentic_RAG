# Agentic RAG System with PDF and Website Integration

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system that allows users to retrieve answers from uploaded PDFs, specified website URLs, or a combination of both. The system uses an intelligent agent to decide whether a query can be answered based on the provided sources or needs to fall back on online searches.

## Key Features

1. **PDF Retrieval**: Upload PDF files and extract information for question answering.
2. **Website Retrieval**: Provide URLs to extract and use content for answering queries.
3. **Combined Query Handling**: Simultaneously process PDFs and URLs to retrieve answers.
4. **Agent Logic**: 
    - First checks if the answer exists in the uploaded PDF.
    - If not found, checks the website content.
    - If unavailable in both, declares the question as outside the RAG database and refrains from answering.
5. **Fallback Search**: If no relevant information is found in the provided data, an online search is used to retrieve relevant context.

## Tech Stack

- **Streamlit**: User interface.
- **PyPDF2**: Extract text from PDF files.
- **BeautifulSoup**: Parse and clean website content.
- **OpenAI API**: Generate embeddings and answer questions.
- **Qdrant**: Vector database for semantic search.
- **DuckDuckGo Search**: Online search fallback for out-of-database queries.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/rajveersinghcse/Agentic_RAG
    cd Agentic_RAG
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Qdrant:**
    - Download and install [Qdrant](https://qdrant.tech/documentation/quick_start/).
    - Start Qdrant on `http://localhost:6333`.

## Usage

1. **Run the Application:**
    ```bash
    streamlit run app.py
    ```

2. **Configure API Key:**
   - Enter your OpenAI API Key in the designated input field in the app.

3. **Upload Data:**
   - Upload PDF files or provide website URLs (comma-separated).
   - Optionally, enable crawling to extract content from all linked pages.

4. **Process Data:**
   - Click "Process and Index Documents" to generate embeddings and store them in the Qdrant database.

5. **Ask Questions:**
   - Enter your question in the input field.
   - The agent determines the source of the answer:
       - Retrieves from PDF if present.
       - Falls back to website if not in PDF.
       - If neither, performs an online search (optional) or states that the question is outside the RAG database.

## Agent Workflow

1. **PDF Search**: If the answer is found in the uploaded PDFs, it is retrieved and displayed.
2. **Website Search**: If the answer is not in PDFs, it searches through the provided website content.
3. **Fallback Search**: If neither source contains the answer, the question is identified as outside the RAG database.

## Configuration

- **OpenAI API Key**: Required for embeddings and question-answering models.
- **Qdrant**: Must be running locally or configured to a remote host in the code.

## Requirements
- Python 3.8+ (I used 3.12.7)
- Valid OpenAI API Key
- Running instance of Qdrant

## Dependencies

- `streamlit`
- `PyPDF2`
- `beautifulsoup4`
- `qdrant-client`
- `tqdm`
- `litellm`
- `duckduckgo_search`
- `langchain_text_splitters`

Install all dependencies with:
```bash
pip install -r requirements.txt
```


## FAQ

### What happens if I upload both PDFs and URLs?
The agent processes both and prioritizes the PDFs. If the answer is not in PDFs, it checks the websites.

### Can it answer questions outside the uploaded data?
No. If the answer isn't in the PDFs or URLs, the agent either performs an online search (if enabled) or states that it can't answer.

### What if the Qdrant server isn't running?
Ensure Qdrant is properly installed and started on `localhost:6333` before indexing documents.

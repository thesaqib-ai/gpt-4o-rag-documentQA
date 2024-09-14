import streamlit as st
import re
import os
import json
import logging
import tempfile
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document as LangChainDocument
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

class DocumentProcessor:
    def get_pages(self, uploaded_file):
        if not uploaded_file:
            logging.error('No file provided.')
            raise ValueError('File is empty or not provided!')
        elif uploaded_file:
            return self._get_pdf_pages(uploaded_file)
        else:
            logging.error('Unsupported file type, Please insert pdf document.')
            raise ValueError('Unsupported File Type, Please insert pdf document.')

    def _get_pdf_pages(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as temp_file:
            chunk_size = 8191
            for chunk in iter(lambda: uploaded_file.read(chunk_size), b""):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        return pages

    def create_embeddings(self, document_pages, uploaded_file):
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    
        # List to hold the document objects for embedding
        document_list = []
    
        # Check if `document_pages` contains a list of pages (PDF)
        for page in document_pages:
            # If page is already a LangChain Document (for PDFs)
            page_split = text_splitter.split_text(page.page_content)
            # Create Document objects for each chunk
            for page_sub_split in page_split:
                metadata = {"source": uploaded_file.name, "page_no": page.metadata["page"] + 1}
                document_obj = LangChainDocument(page_content=page_sub_split, metadata=metadata)
                document_list.append(document_obj)
    
        # Extract the file name without extension and clean it
        file_name = os.path.splitext(uploaded_file.name)[0]
        clean_file_name = re.sub(r'[^A-Za-z0-9_]', '_', file_name)
    
        # Initialize embeddings with the selected model
        embedding = lambda text: get_embedding(text, model='text-embedding-3-small')

        qdrant_url = "https://f6c816ad-c10a-4487-9692-88d5ee23882a.europe-west3-0.gcp.cloud.qdrant.io:6333"
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
        collection_name = clean_file_name
    
        # Create Qdrant vector store
        qdrant = QdrantVectorStore.from_documents(
            document_list,
            embedding,
            url=qdrant_url,
            api_key=QDRANT_API_KEY,
            collection_name=collection_name
        )
                            
        return qdrant


    def generate_response(self, retriever, query_text):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=1,
            max_tokens=1024,
            max_retries=2
        )

        template = """Use the following document to answer the question at the end. Go through the content and look for the answers.
        If you don't find relevant information in the document, just say that Please ask relevant questions!, Don't try to make up an answer.
        

        {context}

        Question: {question} according to this document

        Helpful Answer:"""

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        custom_rag_prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke(query_text)


# Streamlit file uploader
st.set_page_config(page_title='DocQA', layout='wide')

def main():
    st.title('🤖GPT-4o-mini Document QA using RAG')

    with st.sidebar:
        st.title('Hi there!')
        st.markdown('Drop your docs here:')
        uploaded_file = st.file_uploader('Upload a pdf file:', type=['pdf'])
    
    # Initialize chat session in Streamlit if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user's message
    query_text = st.chat_input()

    if uploaded_file:
        if 'qdrant' not in st.session_state:
            document_processor = DocumentProcessor()
            try:
                document_pages = document_processor.get_pages(uploaded_file)
                qdrant = document_processor.create_embeddings(document_pages, uploaded_file)
                st.session_state.qdrant = qdrant
                st.success('Document processed and embeddings created successfully!')
            except Exception as e:
                st.error(f'Error processing document: {str(e)}')
    
    # Check if there's a query and if the Qdrant is available
    if query_text and 'qdrant' in st.session_state:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(query_text)
        st.session_state.chat_history.append({"role": "user", "content": query_text})

        # Generate response from the model without the button
        with st.spinner('Thinking...'):
            try:
                retriever = st.session_state.qdrant.as_retriever()
                document_processor = DocumentProcessor()
                response = document_processor.generate_response(retriever, query_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
    elif not uploaded_file and not query_text:
        st.markdown('Document not yet uploaded.')

if __name__ == "__main__":
    main()

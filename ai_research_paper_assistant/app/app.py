from langchain_community.document_loaders import UnstructuredPDFLoader
import os, shutil, tempfile, asyncio # For creating a dummy file, in a real scenario, you'd have your PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()

# --- Dummy PDF Creation (for demonstration purposes only) ---
# In a real project, you would replace this with your actual PDF path.
# For simplicity, we'll just simulate the *content types* expected.
# You'd need a real PDF file with these elements for the code to run fully.
# Let's assume 'complex_document.pdf' exists and has these elements.
# You might use a tool like LibreOffice or Google Docs to create a sample PDF
# with a table (e.g., 'Name | Age') and some code (e.g., 'def hello(): print("Hello")').
# -----------------------------------------------------------
UPLOADED_DIR = "../uploaded_files"

PERSIST_DIR = "../database/chrome_db"


def saved_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(UPLOADED_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

@st.cache_resource(show_spinner=False)
def load_and_process_documents(file_path, file_name, action):
    

    if file_path is None or not os.path.exists(file_path):
        return None
    
    try:
        if "Summarize" and "Compare two documents" not in action:

            doc_persist_dir = os.path.join(PERSIST_DIR, file_name.replace('.','_').replace(' ', '_').lower())
            if os.path.exists(doc_persist_dir) and os.listdir(doc_persist_dir):
                return 'chunk exist'

        else:

            loader_elements = UnstructuredPDFLoader(file_path, mode="elements")

            raw_documents = []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400
            )

            raw_documents = loader_elements.lazy_load()
                

            chunks = text_splitter.split_documents(raw_documents)

            cleaned_chunks = filter_complex_metadata(chunks)
            return cleaned_chunks
    
    except Exception as e:
        st.error(f"Error Processing PDF: {e}")
        return None

@st.cache_resource(show_spinner=False)
def setup_rag_chain(chunks_rag, persist_dir, file_name):
    if not chunks_rag:
        return None
    
    doc_persist_dir = os.path.join(persist_dir, file_name.replace('.','_').replace(' ', '_').lower())

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(doc_persist_dir) and os.listdir(doc_persist_dir):

        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=doc_persist_dir
        )
    else:
        if os.path.exists(doc_persist_dir):
            shutil.rmtree(doc_persist_dir)
        os.makedirs(doc_persist_dir, exist_ok=True)

        vectorstore = Chroma.afrom_documents(
            documents=chunks_rag,
            embedding=embedding_model,
            persist_directory=doc_persist_dir
        )

        vectorstore.persist()

    

    print("Indexing Completed.......")

    # Step 2: Retrieval

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":6, "lambda_mult": 0.5}
    )

    # Step3: Augmentation
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt_question = PromptTemplate(
        template="""
            You are a AI assistant for research paper.. Answer the user's question based ONLY on the provided context. if you cannot find the answer in the context, politely state that you don't have enough information.

            {context}
            Question: {question}
        """,
        input_variables=['context', 'question']

    )

    question = "What are the key findings about Multimodal talent tables in this document"

    # created parallele chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    # create main chain
    parser = StrOutputParser()
    rag_chain = parallel_chain | prompt_question | llm | parser

    return rag_chain

def query_task(query_text):
    if query_text:
        
        #with st.spinner("processing your query..."):
        try:
            
            response = st.session_state.rag_chain.invoke(query_text)
            st.success("Answer Generated")
            st.write("### Answer")
            

        except Exception as e:
            st.error(f"An Error occurred during query: {e}")
            st.warning("Please Ensure API key is correctly set and document is valid")
    return response


def summarize(documents: list[Document], chain_type: str = "stuff"):
    if not documents:
        return "No Documents available to summarize" 
    max_words = 300
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    

    summary_prompt_template = """
    You are an expert summarizer. Your goal is to create a concise and accurate summary of the following text.
    Focus on the main points, key arguments, and essential information.
    The summary should be easy to understand and no longer than {max_words} words.

    Text:
    "{text}"

    Summary:
    """

    summary_prompt = PromptTemplate(
        template=summary_prompt_template,
        input_variables=["text", "max_words"]
    )
    map_prompt = summary_prompt
    combine_prompt = summary_prompt

    if chain_type == "stuff":
        summary_chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
        


    elif chain_type == "map_reduce":
        summary_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            token_max=400,
            verbose=False
        )

    summary_result = summary_chain.invoke({"input_documents": documents, "max_words": max_words})
    
    return summary_result.get("output_text", "Could not generate Summary.")


def compare_documents(document1_content, document2_content):
    if not (document1_content or document2_content):
        return "Documents are not available for compare!"
    max_words = 500
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    parser = StrOutputParser()
    comparison_prompt_template = """
    You are an expert document analyst specializing in comparing texts.
    Your task is to analyze and compare the two documents provided below no longer than {max_words} words.

    ---
    Document 1: "{document1_content}"
    ---

    ---
    Document 2: "{document2_content}"
    ---

    Please provide a detailed comparison focusing on the following aspects:
    1.  **Main Topics/Themes:** What are the primary subjects discussed in each document, and are there any common or unique themes?
    2.  **Key Arguments/Findings:** What are the most important arguments or findings presented in each document? Are there agreements or disagreements?
    3.  **Differences:** Clearly state the significant differences, contradictions, or unique information present in one document but not the other.
    4.  **Similarities:** Clearly state the significant similarities, overlapping information, or reinforcing points between the two documents.
    5.  **Overall Relationship:** Briefly describe how the two documents relate to each other (e.g., one expands on the other, they present opposing views, one is an update of the other).

    Structure your response clearly with headings for each aspect.
    Comparison:
    """

    comparison_prompt = PromptTemplate(
        template=comparison_prompt_template,
        input_variables=["document1_content", "document2_content", "max_words"]
    )

    chain = comparison_prompt | llm | parser
    result = chain.invoke({"document1_content": document1_content, "document2_content": document2_content, "max_words": max_words})

    return result



def file_upload(uploaded_file):
    if uploaded_file is not None:
        if st.session_state.get("uploaded_file_id") != uploaded_file.file_id:
            saved_file_path = saved_uploaded_file(uploaded_file)
            st.session_state.uploaded_file_path= saved_file_path
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_file_id = uploaded_file.file_id
            #st.session_state.uploaded_file_path = uploaded_file_path
            st.success(f"Document '{uploaded_file.name}' saved and ready for processing ")
            st.session_state.chunks_for_rag = None
            st.session_state.rag_chain = None

    return

def reset_session_var():
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "uploaded_file_id" not in st.session_state:
        st.session_state.uploaded_file_id =None
    if "chunks_for_rag" not in st.session_state:
        st.session_state.chunks_for_rag = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "summary_doc1" not in st.session_state:
        st.session_state.summary_doc1 = None
    if "summary_doc2" not in st.session_state:
        st.session_state.summary_doc2 = None
    if "uploaded_file1" not in st.session_state:
        st.session_state.uploaded_file1 = None
    if "uploaded_file2" not in st.session_state:
        st.session_state.uploaded_file2 = None
    if "doc1_chunks" not in st.session_state:
        st.session_state.doc1_chunks = None
    if "doc2_chunks" not in st.session_state:
        st.session_state.doc2_chunks = None
    if "uploaded_file_path_doc1" not in st.session_state:
        st.session_state.uploaded_file_path_doc1 = None
    if "uploaded_file_path_doc2" not in st.session_state:
        st.session_state.uploaded_file_path_doc2 = None
    if "compare" not in st.session_state:
        st.session_state.compare = None
    
    return


def main():
    st.set_page_config(page_title="Research Paper Assistant") # , layout="wide"
    st.header("AI Document Processing & Analysis")

    
    
    st.subheader("Choose an Action")
    action = st.selectbox( "Select Action", ["Query Document", "Summarize", "Compare two documents"], index=0)
    reset_session_var()
    #implement query document feature now

    if action == "Query Document":
        st.subheader("Upload document & Ask a question")
        uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
        query_text = st.text_input("Enter your question about the document: ", placeholder="e.g., what are the main findings?", key="query_input")
        
        if st.button("Submit"):
            with st.spinner(f"Processing your query... "):
                file_upload(uploaded_file)
                        

                if st.session_state.chunks_for_rag is None and st.session_state.uploaded_file_path:
                        
                    st.session_state.chunks_for_rag = load_and_process_documents(st.session_state.uploaded_file_path, st.session_state.uploaded_file_name, action)

                if st.session_state.chunks_for_rag:
                    if st.session_state.rag_chain is None:
                        st.session_state.rag_chain = setup_rag_chain(
                            st.session_state.chunks_for_rag,
                            PERSIST_DIR,
                            st.session_state.uploaded_file_name
                        )

                    if st.session_state.rag_chain:
                        st.session_state.response = query_task(query_text)
                        st.write(st.session_state.response)
                            
                            
                    else:
                        st.warning("RAG chain could not be initialized. Please check the document and configuration")

                else:
                    st.info("Upload a PDF document above to enable quering.")

            




    elif action == "Summarize":
        
        st.subheader("Upload document for summary")
        uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
        if st.button("Summarize"):
            
            with st.spinner(f"Generating Summary... "):
                file_upload(uploaded_file)

                if st.session_state.chunks_for_rag is None and st.session_state.uploaded_file_path:
                        
                    st.session_state.chunks_for_rag = load_and_process_documents(st.session_state.uploaded_file_path, st.session_state.uploaded_file_name, action)

                if st.session_state.chunks_for_rag:
                    st.session_state.summary = summarize(st.session_state.chunks_for_rag, chain_type="stuff")
                    st.success("Summary Generated")
                    st.write(st.session_state.summary)

                else:
                    st.warning("Please make sure PDF document is Uploaded!")
        
    elif action == "Compare two documents":
        st.subheader("Upload documents for Comparison")
        uploaded_file1 = st.file_uploader("Upload First PDF document", type="pdf")
        uploaded_file2 = st.file_uploader("Upload Second PDF document", type="pdf")

        if st.button("Compare Documents"):

            with st.spinner(f"Comparing Documents..."):
                file_upload(uploaded_file1)
                st.session_state.uploaded_file_path_doc1 = st.session_state.uploaded_file_path
                
                file_upload(uploaded_file2)
                st.session_state.uploaded_file_path_doc2 = st.session_state.uploaded_file_path

                if (st.session_state.doc1_chunks and st.session_state.doc2_chunks) is None and (st.session_state.uploaded_file_path_doc1 and st.session_state.uploaded_file_path_doc2):
                        
                    st.session_state.doc1_chunks = load_and_process_documents(st.session_state.uploaded_file_path_doc1, st.session_state.uploaded_file_name, action)
                    st.session_state.doc2_chunks = load_and_process_documents(st.session_state.uploaded_file_path_doc2, st.session_state.uploaded_file_name, action)

                if st.session_state.doc1_chunks and st.session_state.doc2_chunks:
                    st.session_state.summary_doc1 = summarize(st.session_state.doc1_chunks, chain_type="stuff")
                    st.session_state.summary_doc2 = summarize(st.session_state.doc2_chunks, chain_type="stuff")
                    st.session_state.compare = compare_documents(st.session_state.summary_doc1, st.session_state.summary_doc2)
                    st.success("Check Comparision Below")
                    st.write(st.session_state.compare)

                else:
                    st.warning("Please make sure PDF documents are Uploaded!")               

        


if __name__ == '__main__':
    main()
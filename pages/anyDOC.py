import os
import openai
import sys
from dotenv import load_dotenv
import shutil
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

import streamlit as st
from io import StringIO

# sys.path.append('../..')
database_path = "database/chroma_db"

API_KEY = open('key.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

openai.api_key = API_KEY

from dotenv import load_dotenv
load_dotenv()

# Create the main panel
st.title("anyDOC :bookmark_tabs:")
st.subheader("Upload and chat with your documents!")

uploaded_file = None
# Create the sidebar for DB connection parameters
# st.sidebar.header("Upload Your Documents")
uploaded_file = st.file_uploader('Choose your file', type="pdf")

filename = "uploaded_pdfs/document.pdf"

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    with open(filename, 'wb') as f: 
        f.write(bytes_data)


    # if uploaded_file is not None:

    loaders = [#PyPDFLoader("/home/sohug/DATAPRAME/DATA/pdf/attention.pdf"),
            PyPDFLoader("uploaded_pdfs/document.pdf"),
            #PyPDFLoader("DATA/pdf/AD-IQ SOP.pdf"),
            #PyPDFLoader("DATA/pdf/The-Little-Prince.pdf"),
            ]

    data = []
    for loader in loaders:
        data.extend(loader.load())

    #token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    data_splits = text_splitter.split_documents(data)


    # create the open-source embedding function
    embedding = OpenAIEmbeddings()

    shutil.rmtree(database_path)
    # load it into Chroma
    database = Chroma.from_documents(data_splits, embedding, persist_directory=database_path)
    #print(database._collection.count())


    llm = OpenAI(temperature=0)

    #Compression retriever
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=database.as_retriever()
    )
    #docs = retriever.get_relevant_documents(query)

    # # Build prompt
    # template = """Use the following pieces of context to answer the question at the end. 
    # If you don't know the exact answer, just say that you don't know, don't try to make up an answer. 
    # Use fifty sentences maximum. Keep the answer as concise as possible. 
    # Always say "Thanks for asking!" at the end of the answer. 
    # {context}
    # Question: {question}
    # Helpful Answer:"""

    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=template,
    )


    # # Run chain
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever = compression_retriever,
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    #     #chain_type="refine"
    #     verbose=False,
    # )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        return_source_documents=True,
        retriever=compression_retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferWindowMemory(
                memory_key="history",
                input_key="question",
                k=5),
        }
    )


    # # Create the main panel
    # st.title("anyDoc :bookmark_tabs:")
    # st.subheader("Upload and chat with your documents!")

# Get the user's natural question input
question = st.text_input(":blue[Ask a question:]", placeholder="Enter your question.")

# Create a submit button for executing the query
query_button = st.button("Submit")

# Execute the query when the submit button is clicked
if query_button:

    # if not submit_button:
    #     st.warning(":wave: Please connect to the database first.")
    #     st.stop()

    try:
        with st.spinner(text='Thinking... 🤔'):
            print("\nQuestion: " + str(question))
            # print(str(question))
            result = qa_chain({"query": question})

        st.subheader("Answer :robot_face:")
        st.write(result['result'])


        st.info(":coffee: _Did that answer your question? If not, try to be more specific._")
    except Exception as e:
        print(e)
        st.warning(":wave: Please enter a valid question. Try to be as specific as possible.")
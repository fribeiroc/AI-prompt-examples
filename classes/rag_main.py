'''
This class defines the rag_main class which structures the usage
of RAG (Retrieval-Augmented-Generation) models
'''

import logging

#To load many documents in one run
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Data path that will send all raw data to RAG model
DATA_PATH = "raw_pdf_data"
logging.basicConfig(filename="C:\Programacao\Logs")

class rag_main:
    def load_pdf_documents():
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
        documents = loader.load()
        logging.info(len(documents) + " .pdf documents where loaded.")
        return documents

    def split_doc_data(documents):
        text_splitter = RecursiveCharacterTextSplitter(
           chunk_size = 1000,
           chunk_overlap=500,
           length_function=len
           add_start_index=True,
        )
        return text_splitter.split_documents(documents)

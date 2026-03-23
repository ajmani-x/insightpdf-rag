from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings

load_dotenv()

def text_loader(pdf_path=""):

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    return docs

def split_text(docs):

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)

    split_docs = splitter.split_documents(docs)

    return split_docs

def vector_store(split_docs):

    embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    )

    db = Chroma.from_documents(documents=split_docs,embedding=embeddings,persist_directory='db/chroma_db')

    return db

def main():

    path = "data/google_data.pdf"

    text = text_loader(path)

    split_docs = split_text(text)

    db = vector_store(split_docs)

if __name__ == "__main__":
    main()

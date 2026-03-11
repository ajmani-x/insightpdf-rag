from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings

load_dotenv()

def load_vector_db():

    model = CohereEmbeddings(
        model="embed-english-v3.0"
    )

    db = Chroma(persist_directory='db/chroma_db',embedding_function=model)

    return db

def retrieve_docs(db,query):

    retriever = db.as_retriever(search_kwargs={'k':3})

    retrieve = retriever.invoke(query)

    return retrieve

def gen_result(query,docs):

    model = ChatCohere()

    prompt =  f'''Answer the follwing question with the given context and if the context is not enough to answer the question just say dont know \n Context -> {docs} \n Question -> {query}'''

    result = model.invoke(prompt)

    return result.content

def main():

    db = load_vector_db()

    query = 'When was google found ?'

    docs = retrieve_docs(db,query)

    result = gen_result(query,docs)

    print(result)

if __name__ == "__main__":
    main()



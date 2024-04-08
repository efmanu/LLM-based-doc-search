from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import  llm
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama




if __name__ == '__main__':

    documents = SimpleDirectoryReader("data").load_data()

    from llama_index.llms.ollama import Ollama

    llm = Ollama(model="llama2", request_timeout=30.0)

    # bge embedding model
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="phi", request_timeout=30.0)

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    print("Querying: What did the author do growing up?")
    response = query_engine.query("What did the author do growing up?")
    print(response)
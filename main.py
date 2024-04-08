from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama




if __name__ == '__main__':

    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} docs\n")  # Check the total number of loaded documents

    # bge embedding model
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="phi", request_timeout=300.0)

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    print("Querying: Who is Prithviraj Sukumaran? \n")
    response = query_engine.query("Who is Prithviraj Sukumaran?")
    print(response)
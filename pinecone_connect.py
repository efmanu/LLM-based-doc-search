import os
from pinecone import Pinecone, PodSpec

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore


# initialize without metadata filter
from llama_index.core.storage import StorageContext

from llama_index.embeddings.openai import OpenAIEmbedding


# This will be the model we use both for Node parsing and for vectorization
embed_model = OpenAIEmbedding(api_key="<OPEN_API_KEY>")

api_key = os.environ.get("PINECONE_API_KEY","<PINECONE_API_KEY>")
pc = Pinecone(api_key=api_key)

index_name="quickstart"

# List all indexes
all_indexes = pc.list_indexes()

# Create new index if not exists
if index_name in all_indexes.get('indexes',[]):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=PodSpec(environment='gcp-starter', pod_type='s1.x1'),
    )

pinecone_index = pc.Index(index_name)

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
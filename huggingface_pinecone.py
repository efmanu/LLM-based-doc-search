import os
from pinecone import Pinecone, PodSpec
from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import  Settings
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN", '<HUGGINGFACE_TOKEN>')
remotely_run = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN
)

# locally_run = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.llm = remotely_run
api_key = os.environ.get("PINECONE_API_KEY","<PINECONE_API_KEY>")
pc = Pinecone(api_key=api_key)

index_name="quickstart"

# List all indexes
all_indexes = pc.list_indexes()
all_index_names = [x['name'] for x in all_indexes]

# Create new index if not exists
if index_name not in all_index_names:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='euclidean',
        spec=PodSpec(environment='gcp-starter', pod_type='s1.x1'),
    )

pinecone_index = pc.Index(index_name)

documents = SimpleDirectoryReader("./data/paul_graham").load_data()

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

query_engine = index.as_query_engine()
response = query_engine.query("What I Worked On?")

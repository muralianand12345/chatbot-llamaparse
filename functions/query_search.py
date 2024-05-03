from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore


def load_index(storage_path):
    vector_store = FaissVectorStore.from_persist_dir(storage_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=storage_path
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index

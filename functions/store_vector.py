import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_parse import LlamaParse

import faiss
from llama_index.vector_stores.faiss import FaissVectorStore


# For Large Datasets
def init_faiss(dimensions, nlist=100):
    # IndexFlatL2 - Euclidean distance
    quantizer = faiss.IndexFlatL2(dimensions)
    faiss_index = faiss.IndexIVFFlat(quantizer, dimensions, nlist)

    np.random.seed(1234)
    xb = np.random.random((10000, dimensions)).astype("float32")
    if not faiss_index.is_trained:
        faiss_index.train(xb)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context


# For Small Datasets
# def init_faiss(dimensions):
#     # IndexFlatL2 - Euclidean distance
#     faiss_index = faiss.IndexFlatL2(dimensions)
#     vector_store = FaissVectorStore(faiss_index=faiss_index)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     return storage_context


def load_documents(result_type, documents_path):
    parser = LlamaParse(result_type=result_type)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        input_files=documents_path,
        file_extractor=file_extractor,
    ).load_data()
    return documents


def read_data_folder(folder_path):
    documents_path = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            documents_path.append(os.path.join(folder_path, file))
    return documents_path

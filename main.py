import os
import json
import nest_asyncio
from fastapi import FastAPI, HTTPException
from dotenv import dotenv_values
from llama_index.core import VectorStoreIndex
from starlette.responses import Response

from functions.store_vector import init_faiss, load_documents, read_data_folder
from functions.query_search import load_index

nest_asyncio.apply()

env_config = dotenv_values(".env")
os.environ["LLAMA_CLOUD_API_KEY"] = env_config["LLAMA_CLOUD_API_KEY"]
os.environ["OPENAI_API_KEY"] = env_config["OPENAI_API_KEY"]

app = FastAPI()

# variables
dimensions = 1536
documents_path = read_data_folder("./data")
result_type = "markdown"
storage_path = "./storage"


# load db on start
def load_db():
    try:
        storage_context = init_faiss(dimensions)
        documents = load_documents(result_type, documents_path)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist()
        return index
    except Exception as e:
        if "No such file or directory" in str(e):
            raise Exception(
                "No such file or directory. Please check the path of the documents folder"
            )
        else:
            raise Exception(str(e))


load_db()
# load index on start
index = load_index(storage_path)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/store/db")
def store_db():
    try:
        load_db()
        return {"message": "Database stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
def query_search(query: str):
    try:
        template = """
            You are a chatbot that helps users to find information and predict from a database of documents.
            The user asks a question and you provide the answer.
            If the question is not found in the database, you should reply 'Kindly request questions related to the subject'.
            User asks: "{}"
            Do not answer if the question is not safe for work.
            Reply with the answer in the format of json with the response, reference docs and image link if available else ignore image link.
        """

        template = template.format(query)

        query_engine = index.as_query_engine()
        bot_response = query_engine.query(template)

        response_json = json.loads(bot_response.response)

        return Response(content=json.dumps(response_json), media_type="application/json")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

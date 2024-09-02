import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import os


COLLECTION_NAME = "articles"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_MODEL_MULTI = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_db = None
model = None


def get_vector_db() -> None:
    global vector_db
    if vector_db is None:
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(allow_reset=True),
        )
        vector_db = chroma_client
    return vector_db


def get_sentence_transformer_model(use_multi_language_model=False) -> None:
    global model
    if model is None:
        model_name = (
            SENTENCE_TRANSFORMER_MODEL_MULTI
            if use_multi_language_model
            else SENTENCE_TRANSFORMER_MODEL
        )
        model_path = "./models"
        os.makedirs(model_path, exist_ok=True)

        if not os.path.exists(os.path.join(model_path, model_name)):
            model_instance = SentenceTransformer(model_name)
            model_instance.save(os.path.join(model_path, model_name))
        else:
            model_instance = SentenceTransformer(
                os.path.join(model_path, model_name),
            )

        model = model_instance
    return model

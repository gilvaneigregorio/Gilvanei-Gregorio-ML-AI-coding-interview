import mongoengine as db
from utils import (
    get_vector_db,
    get_sentence_transformer_model,
    COLLECTION_NAME,
)


class Article(db.Document):
    text = db.StringField(required=True)

    # Default lexical search
    @classmethod
    def lexical_search(cls, query, top_k=10) -> list[str]:
        return [a.text for a in cls.objects(text__icontains=query).limit(top_k)]  # noqa

    # First version of semantic search
    @classmethod
    def search(cls, query, top_k=10, similarity_threshold=0.5) -> list[dict]:  # noqa
        model = get_sentence_transformer_model()
        vector_db = get_vector_db()

        query_embedding = model.encode(query).tolist()

        collection = vector_db.get_collection(COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        distances = results["distances"][0]
        documents = results["documents"][0]

        similarity_scores = [1 / (1 + distance) for distance in distances]

        filtered_results = []
        for i in range(len(documents)):
            doc = documents[i]
            score = similarity_scores[i]
            if score >= similarity_threshold:
                filtered_results.append(doc)

        return filtered_results

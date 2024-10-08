import os

import click
import pandas as pd
from flask.cli import with_appcontext
from mongoengine import connect
from search_app.models import Article
from utils import get_vector_db, get_sentence_transformer_model, COLLECTION_NAME


@click.command("load-news")
@with_appcontext
def load_news():
    connect(
        host=os.environ["MONGODB_HOST"],
        db=os.environ["MONGODB_DB"],
    )
    articles = 0
    print("starting load")
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    df = pd.read_parquet("hf://datasets/fancyzhx/ag_news/" + splits["test"])

    model = get_sentence_transformer_model()
    vector_db = get_vector_db()
    collection = vector_db.get_collection(COLLECTION_NAME)
    for article in df.itertuples():
        articles += 1
        Article.objects(text=article.text).update_one(
            upsert=True, set__text=article.text
        )

        embedding = model.encode(article.text).tolist()

        collection.add(
            embeddings=[embedding],
            documents=[article.text],
            ids=[str(articles)],
        )

    print(f"Loaded {articles} articles!")

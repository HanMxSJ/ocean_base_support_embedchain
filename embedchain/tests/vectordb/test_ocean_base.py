import shutil

import pytest
from chromadb import Settings

from embedchain import App
from embedchain.config import AppConfig, BaseEmbedderConfig
from embedchain.config.vector_db.ocean_base import OceanBaseDBConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.embedder.openai import OpenAIEmbedder
from embedchain.vectordb.ocean_base import OceanBaseDB
from unittest.mock import patch

ob_config = OceanBaseDBConfig(
            host= '6.14.140.244',
            port= 31200,
            user= 'test@test',
            password= '',
            db_name= 'test',
            charset= 'utf8mb4',
            collection_name = 'ocean_base_coll',
            embedder=OpenAIEmbedder(
                ##provider="openai",
                config=BaseEmbedderConfig(
                    model="bailing_1b_embedding",
                    api_key='lKvQac6mTlqFFKRkgfscxxt7UvsR7PbU',
                    api_base='https://antchat.alipay.com/v1'
                ),
            ),
        )


@pytest.fixture
def ocean_base_db():
    return OceanBaseDB(config=OceanBaseDBConfig(user='test@test' ,host="6.14.140.244", port=31200 , database='test'))


@pytest.fixture
def app_with_settings():
    ob = OceanBaseDB(config=OceanBaseDBConfig(user='test@test', host="6.14.140.244", port=31200, database='test'))
    app_config = AppConfig(collect_metrics=False)
    return App(config=app_config, db=ob)


@pytest.fixture(scope="session", autouse=True)
def cleanup_db():
    yield
    try:
        shutil.rmtree("test-db")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def test_ocean_base_db_duplicates_throw_warning(caplog):
    db = OceanBaseDB(config=ob_config)
    app = App(embedding_model = ob_config.embedder,config=AppConfig(collect_metrics=False), db=db)

    documents = ["This is a document.", "This is another document."]
    metadatas = [{"url": "url_1", "doc_id": "doc_id_1"}, {"url": "url_2", "doc_id": "doc_id_2"}]
    ids = ["doc_1", "doc_2"]

    app.db.add(documents =  documents,
                       metadatas = metadatas,
                       ids = ids,
            )
    #app.add(embeddings=[[0, 0, 0]], ids=["0"])
    #app.db.reset()
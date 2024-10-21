import shutil

import pytest
from chromadb import Settings
import unittest
from embedchain import App
from embedchain.config import AppConfig, BaseEmbedderConfig
from embedchain.config.vector_db.ocean_base import OceanBaseDBConfig
from embedchain.embedder.base import BaseEmbedder
from embedchain.embedder.openai import OpenAIEmbedder
from embedchain.vectordb.ocean_base import OceanBaseDB
from unittest.mock import patch

class TestEsDB(unittest.TestCase):

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
                        api_base='https://antchat.alipay.com/v1',
                        #vector_dimension=100,

                    ),
                ),
            )


    @pytest.fixture
    def ocean_base_db(self):
        return OceanBaseDB(config=OceanBaseDBConfig(user='test@test' ,host="6.14.140.244", port=31200 , database='test'))


    @pytest.fixture
    def app_with_settings(self):
        ob = OceanBaseDB(config=OceanBaseDBConfig(user='test@test', host="6.14.140.244", port=31200, database='test'))
        app_config = AppConfig(collect_metrics=False)
        return App(config=app_config, db=ob)


    @pytest.fixture(scope="session", autouse=True)
    def cleanup_db(self):
        yield
        try:
            shutil.rmtree("test-db")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    def test_ocean_base_db_add_query_document(self):
        db = OceanBaseDB(config=self.ob_config)
        app = App(embedding_model = self.ob_config.embedder,config=AppConfig(collect_metrics=False), db=db)

        documents = ["This is a document.", "This is another document."]
        metadatas = [{"url": "url_1", "doc_id": "doc_id_1"}, {"url": "url_2", "doc_id": "doc_id_2"}]
        ids = ["doc_1", "doc_2"]

        app.db.reset()
        print('count_1:' ,app.db.count())
        assert app.db.count() == 0

        app.db.add(documents =  documents,
                           metadatas = metadatas,
                           ids = ids,
                )

        query = "This is a document"
        results_without_citations = app.db.query(query, n_results=1,)
        print(results_without_citations)

        print('count_2:',app.db.count())
        assert app.db.count() == 2

    def test_ocean_base_db_query_document(self):
        db = OceanBaseDB(config=self.ob_config)
        app = App(embedding_model = self.ob_config.embedder,config=AppConfig(collect_metrics=False), db=db)

        query = "This is a document"
        results_without_citations = app.db.query(query, n_results=2,)
        print(results_without_citations)
        expected_results_without_citations = [('This is a document.', {'doc_id': 'doc_id_1', 'url': 'url_1'}),
                                                ('This is another document.', {'doc_id': 'doc_id_2', 'url': 'url_2'})]
        self.assertEqual(results_without_citations, expected_results_without_citations)

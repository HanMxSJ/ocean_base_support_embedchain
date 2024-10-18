import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import pymysql
import uuid
import json
import threading

from langchain_core.documents import Document
from nltk.corpus.reader import documents
from sqlalchemy.types import UserDefinedType, Float, String
from sqlalchemy import Column, String, Table, create_engine, insert, text
from sqlalchemy.dialects.mysql import JSON, LONGTEXT, VARCHAR
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from embedchain.helpers.json_serializable import register_deserializable
from embedchain.vectordb.base import BaseVectorDB
from embedchain.config.vector_db.ocean_base import OceanBaseDBConfig

from tqdm import tqdm

_EMBEDCHAIN_OCEANBASE_DEFAULT_EMBEDDING_DIM = 1536
_EMBEDCHAIN_OCEANBASE_DEFAULT_COLLECTION_NAME = "embedchain_document"
_EMBEDCHAIN_OCEANBASE_DEFAULT_IVFFLAT_CREATION_ROW_THRESHOLD = 10000
_EMBEDCHAIN_OCEANBASE_DEFAULT_RWLOCK_MAX_READER = 64

logger = logging.getLogger(__name__)


Base = declarative_base()

def from_db(value):
    return [float(v) for v in value[1:-1].split(',')]

def to_db(value, dim=None):
    if value is None:
        return value

    return '[' + ','.join([str(float(v)) for v in value]) + ']'


class Vector(UserDefinedType):
    cache_ok = True
    _string = String()

    def __init__(self, dim):
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        return "VECTOR(%d)" % self.dim

    def bind_processor(self, dialect):
        def process(value):
            return to_db(value, self.dim)
        return process

    def literal_processor(self, dialect):
        string_literal_processor = self._string._cached_literal_processor(dialect)

        def process(value):
            return string_literal_processor(to_db(value, self.dim))
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            return from_db(value)
        return process

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other):
            return self.op('<->', return_type=Float)(other)

        def max_inner_product(self, other):
            return self.op('<#>', return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op('<=>', return_type=Float)(other)


class OceanBaseGlobalRWLock:
    def __init__(self, max_readers) -> None:
        self.max_readers_ = max_readers
        self.writer_entered_ = False
        self.reader_cnt_ = 0
        self.mutex_ = threading.Lock()
        self.writer_cv_ = threading.Condition(self.mutex_)
        self.reader_cv_ = threading.Condition(self.mutex_)

    def rlock(self):
        self.mutex_.acquire()
        while self.writer_entered_ or self.max_readers_ == self.reader_cnt_:
            self.reader_cv_.wait()
        self.reader_cnt_ += 1
        self.mutex_.release()

    def runlock(self):
        self.mutex_.acquire()
        self.reader_cnt_ -= 1
        if self.writer_entered_:
            if 0 == self.reader_cnt_:
                self.writer_cv_.notify(1)
        else:
            if self.max_readers_ - 1 == self.reader_cnt_:
                self.reader_cv_.notify(1)
        self.mutex_.release()

    def wlock(self):
        self.mutex_.acquire()
        while self.writer_entered_:
            self.reader_cv_.wait()
        self.writer_entered_ = True
        while 0 < self.reader_cnt_:
            self.writer_cv_.wait()
        self.mutex_.release()

    def wunlock(self):
        self.mutex_.acquire()
        self.writer_entered_ = False
        self.reader_cv_.notifyAll()
        self.mutex_.release()

    class OBRLock:
        def __init__(self, rwlock) -> None:
            self.rwlock_ = rwlock

        def __enter__(self):
            self.rwlock_.rlock()

        def __exit__(self, exc_type, exc_value, traceback):
            self.rwlock_.runlock()

    class OBWLock:
        def __init__(self, rwlock) -> None:
            self.rwlock_ = rwlock

        def __enter__(self):
            self.rwlock_.wlock()

        def __exit__(self, exc_type, exc_value, traceback):
            self.rwlock_.wunlock()

    def reader_lock(self):
        return self.OBRLock(self)

    def writer_lock(self):
        return self.OBWLock(self)


ob_grwlock = OceanBaseGlobalRWLock(_EMBEDCHAIN_OCEANBASE_DEFAULT_RWLOCK_MAX_READER)

@register_deserializable
class OceanBaseDB(BaseVectorDB):
    """Vector database using OceanBaseDB."""

    def __init__(self, config: Optional[OceanBaseDBConfig] = None):
        """Initialize a new OceanBaseDB instance

        :param config: Configuration options for OceanBaseDB, defaults to None
        :type config: Optional[OceanBaseDbConfig], optional
        """
        if config:
            self.config = config
        else:
            self.config = OceanBaseDBConfig()

        # 将config中的内容直接传递给pymysql.connect()
        self.conn = pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset=self.config.charset
        )

        # Remove auth credentials from config after successful connection
        super().__init__(config=self.config)

    def _initialize(self):
        """
        This method is needed because `embedder` attribute needs to be set externally before it can be initialized.
        """
        if not self.embedder:
            raise ValueError(
                "Embedder not set. Please set an embedder with `_set_embedder()` function before initialization."
            )
        self._get_or_create_collection(self.config.collection_name)

    def _get_or_create_db(self):
        """Called during initialization"""
        return self.conn



    def _get_or_create_collection(self, name):
        """
               Get or create a named collection.

               :param name: Name of the collection
               :type name: str
               :raises ValueError: No embedder configured.
               :return: Created collection
               :rtype: Collection
               """
        if not hasattr(self, "embedder") or not self.embedder:
            raise ValueError("Cannot create a Chroma database collection without an embedder.")
        self.collection = self.conn
        return self.collection


    def get(self, ids: Optional[list[str]] = None, where: Optional[dict[str, any]] = None, limit: Optional[int] = None):
        """
                Get existing doc ids present in vector database

                :param ids: list of doc ids to check for existence
                :type ids: list[str]
                :param where: Optional. to filter data
                :type where: dict[str, Any]
                :param limit: Optional. maximum number of documents
                :type limit: Optional[int]
                :return: Existing documents.
                :rtype: list[str]
                """
        # Define the table schema
        chunks_table = Table(
            self.config.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", LONGTEXT, nullable=True),
            Column("metadata", JSON, nullable=True),  # filter
            keep_existing=True,
        )

        try:
            with self.conn.connect() as conn:
                with conn.begin():
                    query_condition = chunks_table.c.id.in_(ids)
                    query_documents = query_condition.get("document")
                    return query_documents
        except Exception as e:
            self.logger.error("Delete operation failed:", str(e))
            return False



    def add(
            self,
            documents: list[str],
            metadatas: list[object],
            ids: list[str],
            **kwargs: Optional[dict[str, any]],
    ) -> Any:
        embeddings = self.embedder.embedding_fn(documents)
        if len(embeddings) == 0:
            return ids

        if not metadatas:
            metadatas = [{} for _ in documents]

        if self.delay_table_creation and (
                self.collection_stat is None or self.collection_stat.get_maybe_collection_not_exist()):
            self.embedding_dimension = len(embeddings[0])
            self.create_table_if_not_exists()
            self.delay_table_creation = False
            if self.collection_stat is not None:
                self.collection_stat.collection_exists()

        chunks_table = Table(
            self.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", LONGTEXT, nullable=True),
            Column("metadata", JSON, nullable=True),  # filter
            keep_existing=True,
        )

        row_count_query = f"""
                   SELECT COUNT(*) as count FROM `{self.collection_name}`
               """
        chunks_table_data = []
        # try:
        with self.engine.connect() as conn:
            with conn.begin():
                for document, metadata, chunk_id, embedding in zip(
                        documents, metadatas, ids, embeddings
                ):
                    chunks_table_data.append(
                        {
                            "id": chunk_id,
                            "embedding": embedding,
                            "document": document,
                            "metadata": metadata,
                        }
                    )

                    # Execute the batch insert when the batch size is reached
                    if len(chunks_table_data) == self.config.batch_size:
                        with ob_grwlock.reader_lock():
                            if self.sql_logger is not None:
                                insert_sql_for_log = str(insert(chunks_table).values(chunks_table_data))
                                self.sql_logger.debug(
                                    f"""Trying to insert vectors: 
                                               {insert_sql_for_log}""")
                            conn.execute(insert(chunks_table).values(chunks_table_data))
                        # Clear the chunks_table_data list for the next batch
                        chunks_table_data.clear()

                # Insert any remaining records that didn't make up a full batch
                if chunks_table_data:
                    with ob_grwlock.reader_lock():
                        if self.sql_logger is not None:
                            insert_sql_for_log = str(insert(chunks_table).values(chunks_table_data))
                            self.sql_logger.debug(
                                f"""Trying to insert vectors: 
                                           {insert_sql_for_log}""")
                        conn.execute(insert(chunks_table).values(chunks_table_data))

                # if self.sql_logger is not None:
                #     self.sql_logger.debug(f"Get the number of vectors: {row_count_query}")
                if self.enable_index and (
                        self.collection_stat is None or self.collection_stat.get_maybe_collection_index_not_exist()):
                    with ob_grwlock.reader_lock():
                        row_cnt_res = conn.execute(text(row_count_query))
                    for row in row_cnt_res:
                        if row.count > self.th_create_ivfflat_index:
                            self.create_collection_ivfflat_index_if_not_exists()
                            if self.collection_stat is not None:
                                self.collection_stat.collection_index_exists()

        # except Exception as e:
        #     print(f"OceanBase add_text failed: {str(e)}")

        return ids



    def query(
            self,
            input_query: str,
            n_results: Optional[int] = 4,
            where: Optional[dict[str, any]] = None,
            citations: bool = False,
            **kwargs: Optional[dict[str, Any]],
    ) -> Union[list[tuple[str, dict]], list[str]]:
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )

        # filter is not support in OceanBase.

        embedding_str = to_db(self.embedder, self.embedding_dimension)
        sql_query = f"""
            SELECT document, metadata, embedding <-> '{embedding_str}' as distance
            FROM {self.collection_name}
            ORDER BY embedding <-> '{embedding_str}'
            LIMIT :n_results
        """
        sql_query_str_for_log = f"""
            SELECT document, metadata, embedding <-> '?' as distance
            FROM {self.collection_name}
            ORDER BY embedding <-> '?'
            LIMIT {n_results}
        """

        params = {"k": n_results}
        try:
            with ob_grwlock.reader_lock():
                with self.engine.connect() as conn:
                    if self.sql_logger is not None:
                        self.sql_logger.debug(f"Trying to do similarity search: {sql_query_str_for_log}")
                    results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()

            documents_with_scores = [
                (
                    Document(
                        page_content=result.document,
                        metadata=json.loads(result.metadata),
                    ),
                    result.distance if self.embedder is not None else None,
                )
                for result in results
            ]
            contexts = []
            for doc, score in documents_with_scores:
                context = doc.page_content
                if citations:
                    metadata = doc.metadata
                    metadata["score"] = score
                    contexts.append(tuple((context, metadata)))
                else:
                    contexts.append(context)
            return contexts
        except Exception as e:
            self.logger.error("similarity_search_with_score_by_vector failed:", str(e))
            return []

    def count(self) -> int:
        """
        Count number of documents/chunks embedded in the database.

        :return: number of documents
        :rtype: int
        """
        query = {"query": {"match_all": {}}}
        response = self.client.count(index=self.config.collection_name, body=query)
        doc_count = response["count"]
        return doc_count

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        """
        if self.client.indices.exists(index=self.config.collection_name):
            # delete index in ES
            self.client.indices.delete(index=self.config.collection_name)

    def set_collection_name(self, name: str):
        """
        Set the name of the collection. A collection is an isolated space for vectors.

        :param name: Name of the collection.
        :type name: str
        """
        if not isinstance(name, str):
            raise TypeError("Collection name must be a string")
        self.config.collection_name = name
        self._get_or_create_collection(self.config.collection_name)

    def delete(self,ids: Optional[List[str]] = None):
        if ids is None:
            raise ValueError("No ids provided to delete.")

        # Define the table schema
        chunks_table = Table(
            self.config.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", LONGTEXT, nullable=True),
            Column("metadata", JSON, nullable=True),  # filter
            keep_existing=True,
        )

        try:
            with self.conn.connect() as conn:
                with conn.begin():
                    delete_condition = chunks_table.c.id.in_(ids)
                    delete_stmt = chunks_table.delete().where(delete_condition)
                    with ob_grwlock.reader_lock():
                        if self.sql_logger is not None:
                            self.sql_logger.debug(f"Trying to delete vectors: {str(delete_stmt)}")
                        conn.execute(delete_stmt)
                    return True
        except Exception as e:
            self.logger.error("Delete operation failed:", str(e))
            return False
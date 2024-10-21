import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
from sqlalchemy import JSON, Column, String, Table, func, text
import traceback
import uuid
import json
import threading
import numpy as np

try:
    from pyobvector import ObVecClient
except ImportError:
    raise ImportError(
        "Could not import pyobvector package. "
        "Please install it with `pip install pyobvector`."
    )
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

DEFAULT_OCEAN_BASE_VECTOR_TABLE_NAME = "embedchain_vector"
DEFAULT_OCEAN_BASE_HNSW_BUILD_PARAM = {"M": 16, "efConstruction": 256}
DEFAULT_OCEAN_BASE_HNSW_SEARCH_PARAM = {"efSearch": 64}
OCEAN_BASE_SUPPORTED_VECTOR_INDEX_TYPE = "HNSW"
DEFAULT_OCEAN_BASE_VECTOR_METRIC_TYPE = "l2"

DEFAULT_METADATA_FIELD = "metadata"

logger = logging.getLogger(__name__)



@register_deserializable
class OceanBaseDB(BaseVectorDB):
    """Vector database using OceanBaseDB."""

    def __init__(self,
                 config: Optional[OceanBaseDBConfig] = None, #连接配置
                 table_name: str = DEFAULT_OCEAN_BASE_VECTOR_TABLE_NAME, #表名
                 vidx_metric_type: str = DEFAULT_OCEAN_BASE_VECTOR_METRIC_TYPE,
                 vidx_algo_params: Optional[dict] = None,
                 drop_old: bool = False,
                 primary_field: str = "id",
                 vector_field: str = "embedding",
                 text_field: str = "document",
                 metadata_field: Optional[str] = DEFAULT_METADATA_FIELD,
                 vidx_name: str = "vidx",
                 partitions: Optional[Any] = None,
                 extra_columns: Optional[List[Column]] = None,
                 normalize: bool = False,
                 **kwargs,
                 ):
        """Initialize a new OceanBaseDB instance
        """
        self.config = config
        self.table_name = table_name
        self.connection_args = config
        self.extra_columns = extra_columns
        self.normalize = normalize
        self._create_client(**kwargs)
        assert self.ob_vector is not None

        self.vidx_metric_type = vidx_metric_type.lower()
        if self.vidx_metric_type not in ("l2", "inner_product"):
            raise ValueError(
                "`vidx_metric_type` should be set in `l2`/`inner_product`."
            )

        self.vidx_algo_params = (
            vidx_algo_params
            if vidx_algo_params is not None
            else DEFAULT_OCEAN_BASE_HNSW_BUILD_PARAM
        )

        self.drop_old = drop_old
        self.primary_field = primary_field
        self.vector_field = vector_field
        self.text_field = text_field
        self.metadata_field = metadata_field or DEFAULT_METADATA_FIELD
        self.vidx_name = vidx_name
        self.partition = partitions
        self.hnsw_ef_search = -1

        # Remove auth credentials from config after successful connection
        super().__init__(config=self.config)
        self.embedding_function = config.embedder.embedding_fn


    def _create_client(self, **kwargs):  # type: ignore[no-untyped-def]
        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        host = self.connection_args.host
        port = self.connection_args.port
        user = self.connection_args.user
        password = self.connection_args.password
        db_name = self.connection_args.db_name

        self.ob_vector = ObVecClient(
            uri=host + ":" + str(port),
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )
        self.collection = self.ob_vector

    def _normalize_str(self, vector: str) -> List[str]:
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        arr = arr / norm
        return arr.tolist()

    def _normalize_list_float(self, vector: List[float]) -> List[float]:
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        arr = arr / norm
        return arr.tolist()

    def _load_table(self) -> None:
        table = Table(
            self.table_name,
            self.ob_vector.metadata_obj,
            autoload_with=self.ob_vector.engine,
        )
        column_names = [column.name for column in table.columns]
        optional_len = len(self.extra_columns or []) + 1
        assert len(column_names) == (3 + optional_len)

        logging.info(f"load exist table with {column_names} columns")
        self.primary_field = column_names[0]
        self.vector_field = column_names[1]
        self.text_field = column_names[2]
        self.metadata_field = column_names[3]

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
        return self.collection



    def _get_or_create_collection(self, name:Optional[str] = None):
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
        return self.collection


    def get(self, ids: Optional[list[str]] = None,
            where: Optional[dict[str, any]] = None,
            limit: Optional[int] = None):
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
        """Get entities by vector ID.

                Args:
                    ids (Optional[List[str]]): List of ids to get.

                Returns:
                    List[Document]: Document results for search.
                """
        res = self.ob_vector.get(
            table_name=self.table_name,
            ids=ids,
        )
        return [
            Document(
                page_content=r[0],
                metadata=json.loads(r[1]),
            )
            for r in res.fetchall()
        ]

    def _create_table_with_index(self, embeddings: list) -> None:
        try:
            from pyobvector import VECTOR
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        if self.collection.check_table_exists(self.table_name):
            self._load_table()
            return

        dim = len(embeddings[0])
        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.vector_field, VECTOR(dim)),
            Column(self.text_field, LONGTEXT),
            Column(self.metadata_field, JSON),
        ]
        if self.extra_columns is not None:
            cols.extend(self.extra_columns)

        vidx_params = self.ob_vector.prepare_index_params()
        vidx_params.add_index(
            field_name=self.vector_field,
            index_type=OCEAN_BASE_SUPPORTED_VECTOR_INDEX_TYPE,
            index_name=self.vidx_name,
            metric_type=self.vidx_metric_type,
            params=self.vidx_algo_params,
        )

        self.ob_vector.create_table_with_index_params(
            table_name=self.table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            partitions=self.partition,
        )

    def add(
            self,
            add_documents: Optional[list[str]] = None,
            metadatas: Optional[list[object]] = None,
            ids: Optional[list[str]]=None,
            **kwargs: Optional[dict[str, any]],
    ) -> Any:
        embeddings = self.embedding_function(add_documents)
        if len(embeddings) == 0:
            return ids

        total_count = len(embeddings)
        if total_count == 0:
            return []

        self._create_table_with_index(embeddings)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in add_documents]

        if not metadatas:
            metadatas = [{} for _ in add_documents]

        extra_data = add_documents or [{} for _ in add_documents]

        pks: list[str] = []
        for i in range(0, total_count, self.config.batch_size):
            data = [
                {
                    self.primary_field: index_id,
                    self.vector_field: (
                        embedding if not self.normalize else self._normalize_list_float(embedding)
                    ),
                    self.text_field: document,
                    self.metadata_field: metadata,
                    **extra,
                }
                for index_id, embedding, document, metadata, extra in zip(
                    ids[i: i + self.config.batch_size],
                    embeddings[i: i + self.config.batch_size],
                    add_documents[i: i + self.config.batch_size],
                    metadatas[i: i + self.config.batch_size],
                    extra_data[i: i + self.config.batch_size],
                )
            ]
            try:
                self.ob_vector.insert(
                    table_name=self.table_name,
                    data=data,
                )
                pks.extend(ids[i: i + self.config.batch_size])
            except Exception:
                traceback.print_exc()
                logger.error(
                    f"Failed to insert batch starting at entity:[{i}, {i + self.config.batch_size})"
                )
        return pks



    def query(
            self,
            input_query: Optional[str] = None,
            n_results: Optional[int] = 4,
            param: Optional[dict] = None,
            fltr: Optional[str] = None,
            **kwargs: Any,
    ) -> Union[list[tuple[str, dict]], list[str]]:
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )

        # filter is not support in OceanBase.
        if n_results < 0:
            return []
        if input_query is None:
            return []
        query_vector = self.embedder.to_embeddings(input_query)
        sadf = self.similarity_search_by_vector(
            embedding=query_vector, k=n_results, param=param, fltr=fltr, **kwargs
        )
        return None

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEAN_BASE_HNSW_SEARCH_PARAM`
                for example.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        if k < 0:
            return []

        search_param = (
            param if param is not None else DEFAULT_OCEAN_BASE_HNSW_SEARCH_PARAM
        )
        ef_search = search_param.get(
            "efSearch", DEFAULT_OCEAN_BASE_HNSW_SEARCH_PARAM["efSearch"]
        )
        if ef_search != self.hnsw_ef_search:
            self.ob_vector.set_ob_hnsw_ef_search(ef_search)
            self.hnsw_ef_search = ef_search

        res = self.ob_vector.ann_search(
            table_name=self.table_name,
            vec_data=(embedding if not self.normalize else self._normalize_list_float(embedding)),
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            topk=k,
            output_column_names=[self.text_field, self.metadata_field],
            where_clause=([text(fltr)] if fltr is not None else None),
            **kwargs,
        )
        return [
            Document(
                page_content=r[0],
                metadata=json.loads(r[1]),
            )
            for r in res.fetchall()
        ]

    def _parse_metric_type_str_to_dist_func(self) -> Any:
        if self.vidx_metric_type == "l2":
            return func.l2_distance
        if self.vidx_metric_type == "cosine":
            return func.cosine_distance
        if self.vidx_metric_type == "inner_product":
            return func.negative_inner_product
        raise ValueError(f"Invalid vector index metric type: {self.vidx_metric_type}")

    def count(self) -> int:
        """
        Count number of documents/chunks embedded in the database.

        :return: number of documents
        :rtype: int
        """
        res = self.ob_vector.get(
            table_name=self.table_name,
        )
        return res.rowcount()

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        """
        if self.client.indices.exists(index=self.config.collection_name):
            # delete index in ocean_base
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

    def delete(  # type: ignore[no-untyped-def]
            self,
            ids: Optional[List[str]] = None,
            where: Optional[str] = None,
            **kwargs
    ):
        """Delete by vector ID or boolean expression.

        Args:
            ids (Optional[List[str]]): List of ids to delete.
            where (Optional[str]): Boolean filter that specifies the entities to delete.
        """
        self.ob_vector.delete(
            table_name=self.table_name,
            ids=ids,
            where_clause=([text(where)] if where is not None else None),
        )
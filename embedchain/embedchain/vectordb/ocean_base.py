import logging
from typing import Any, Optional, Union

import pymysql
from embedchain.helpers.json_serializable import register_deserializable
from embedchain.vectordb.base import BaseVectorDB

from embedchain.config.vector_db.ocean_base import OceanBaseDBConfig

logger = logging.getLogger(__name__)


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
        return self.client



    def _get_or_create_collection(self, name):
        """Note: nothing to return here. Discuss later"""



    def get(self, ids: Optional[list[str]] = None, where: Optional[dict[str, any]] = None, limit: Optional[int] = None):
        return None

    def add(
            self,
            documents: list[str],
            metadatas: list[object],
            ids: list[str],
            **kwargs: Optional[dict[str, any]],
    ) -> Any:
        return None



    def query(
            self,
            input_query: str,
            n_results: int,
            where: dict[str, any],
            citations: bool = False,
            **kwargs: Optional[dict[str, Any]],
    ) -> Union[list[tuple[str, dict]], list[str]]:
        return None

    def count(self) -> int:
        """
        Count number of documents/chunks embedded in the database.

        :return: number of documents
        :rtype: int
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        """
        raise NotImplementedError

    def set_collection_name(self, name: str):
        """
        Set the name of the collection. A collection is an isolated space for vectors.

        :param name: Name of the collection.
        :type name: str
        """
        raise NotImplementedError

    def delete(self):
        """Delete from database."""

        raise NotImplementedError
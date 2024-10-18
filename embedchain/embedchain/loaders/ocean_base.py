import hashlib
import logging
from typing import Any, Optional
import pymysql

from embedchain.config.vector_db.ocean_base import OceanBaseDBConfig
from embedchain.loaders.base_loader import BaseLoader
from embedchain.utils.misc import clean_string

logger = logging.getLogger(__name__)


class OceanBaseLoader(BaseLoader):
    def __init__(self, config: Optional[dict[str, Any]]):
        super().__init__()
        if not config:
            raise ValueError(
                f"Invalid sql config: {config}.",
                # todo sanji 需要提供 样例+说明文档
                "Provide the correct config, refer `https://docs.embedchain.ai/data-sources/`.",
            )

        self.config = config
        ob_config = OceanBaseDBConfig(config.get("user"), config.get("host"), config.get("port"), config.get("password"), config.get("database"))
        self.connection = pymysql.connect(ob_config);
        self.cursor = None
        self._setup_loader(config=config)

    def _setup_loader(self, config: dict[str, Any]):
        # todo sanji 设置连接池
        try:
            self.connection = None
            self.cursor = self.connection.cursor()
        except (Exception, IOError) as err:
            logger.info(f"Connection failed: {err}")
            raise ValueError(
                f"Unable to connect with the given config: {config}.",
            )

    @staticmethod
    def _check_query(query):
        if not isinstance(query, str):
            raise ValueError(
                f"Invalid postgres query: {query}. Provide the valid source to add from postgres, make sure you are following `https://docs.embedchain.ai/data-sources/postgres`",
                # noqa:E501
            )

    def load_data(self, query):
        self._check_query(query)
        try:
            data = []
            data_content = []
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            for result in results:
                doc_content = str(result)
                data.append({"content": doc_content, "meta_data": {"url": query}})
                data_content.append(doc_content)
            doc_id = hashlib.sha256((query + ", ".join(data_content)).encode()).hexdigest()
            return {
                "doc_id": doc_id,
                "data": data,
            }
        except Exception as e:
            raise ValueError(f"Failed to load data using query={query} with: {e}")

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
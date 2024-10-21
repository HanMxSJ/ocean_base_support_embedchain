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
        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        self.config = config
        ob_config = OceanBaseDBConfig(user=config.get("user"), host=config.get("host"), posrt=config.get("port"), password=config.get("password"), db_name=config.get("db_name"))
        if not ob_config.user or not ob_config.host or not ob_config.port or not ob_config.password or not ob_config.db_name:
            raise ValueError("All fields in OceanBaseDBConfig must be provided")

        self.ob_vector = ObVecClient(
            uri=ob_config.host + ":" + str(ob_config.port),
            user=ob_config.user,
            password=ob_config.password,
            db_name=ob_config.db_name,
        )
        self.connection = self.ob_vector

    def close_connection(self):
        if self.connection:
            self.connection = None
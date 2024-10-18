from typing import Optional


from embedchain.embedder.base import BaseEmbedder
from embedchain.config.vector_db.base import BaseVectorDbConfig
from embedchain.helpers.json_serializable import register_deserializable

@register_deserializable
class OceanBaseDBConfig(BaseVectorDbConfig):
    def __init__(
        self,
        user: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        batch_size: Optional[int] = 100,
        charset: Optional[str] = 'utf8mb4',
        allow_reset=False,
        embedder:Optional[BaseEmbedder] = None
    ):
        self.user = user
        self.host = host
        self.port = port
        self.password = password
        self.batch_size = batch_size
        self.charset = charset
        self.allow_reset = allow_reset
        self.batch_size = batch_size
        self.embedder = embedder
        self.database = database

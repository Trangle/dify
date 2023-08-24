import json
import logging
import requests

from typing import (
    List,
    Optional,
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import embed_with_retry, async_embed_with_retry

logger = logging.getLogger(__name__)

class XOpenAIEmbeddings(OpenAIEmbeddings):
    rest_api: str = "http://124.71.148.73/llm/api"
    """api to get infomations"""

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        response = embed_with_retry(
            self,
            input=texts,
            **self._invocation_params,
        )
        batched_embeddings.extend(r["embedding"] for r in response["data"])
        return batched_embeddings

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        response = await async_embed_with_retry(
            self,
            input=texts,
            **self._invocation_params,
        )
        batched_embeddings.extend(r["embedding"] for r in response["data"])
        return batched_embeddings

    def get_num_tokens(self, text: str) -> int:
        token_num_api = self.rest_api + '/v1/token_nums'
        payload = {
            "model": self.model,
            "max_tokens": 512
        }
        if text:
            payload["prompt"] = text
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(token_num_api, data=json.dumps(payload), headers=headers)

        return int(response.json()["tokenCount"])

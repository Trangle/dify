import decimal
import logging

import openai
import tiktoken
from langchain.embeddings import OpenAIEmbeddings

from core.model_providers.error import LLMBadRequestError, LLMAuthorizationError, LLMRateLimitError, \
    LLMAPIUnavailableError, LLMAPIConnectionError
from core.model_providers.models.embedding.base import BaseEmbedding
from core.model_providers.providers.base import BaseModelProvider


class XOpenAIEmbedding(BaseEmbedding):
    def __init__(self, model_provider: BaseModelProvider, name: str):
        self.credentials = model_provider.get_model_credentials(
            model_name=name,
            model_type=self.type
        )

        client = OpenAIEmbeddings(
            max_retries=1,
            **self.credentials
        )

        super().__init__(model_provider, client, name)
    
    @property
    def base_model_name(self) -> str:
        """
        get base model name (not deployment)
        
        :return: str
        """
        return self.credentials.get("base_model_name")

    def get_num_tokens(self, text: str) -> int:
        """
        get num tokens of text.

        :param text:
        :return:
        """
        if len(text) == 0:
            return 0

        enc = tiktoken.encoding_for_model(self.credentials.get('base_model_name'))

        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if isinstance(ex, openai.error.InvalidRequestError):
            logging.warning("Invalid request to XOpenAI API.")
            return LLMBadRequestError(str(ex))
        elif isinstance(ex, openai.error.APIConnectionError):
            logging.warning("Failed to connect to XOpenAI API.")
            return LLMAPIConnectionError(ex.__class__.__name__ + ":" + str(ex))
        elif isinstance(ex, (openai.error.APIError, openai.error.ServiceUnavailableError, openai.error.Timeout)):
            logging.warning("XOpenAI service unavailable.")
            return LLMAPIUnavailableError(ex.__class__.__name__ + ":" + str(ex))
        elif isinstance(ex, openai.error.RateLimitError):
            return LLMRateLimitError('X ' + str(ex))
        elif isinstance(ex, openai.error.AuthenticationError):
            return LLMAuthorizationError('X ' + str(ex))
        elif isinstance(ex, openai.error.OpenAIError):
            return LLMBadRequestError('X ' + ex.__class__.__name__ + ":" + str(ex))
        else:
            return ex

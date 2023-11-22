import decimal
import logging
import traceback
from typing import List, Optional, Any

import openai
from langchain.callbacks.manager import Callbacks
from langchain.schema import LLMResult

from core.model_providers.providers.base import BaseModelProvider
from core.third_party.langchain.llms.chat_xopen_ai import EnhanceChatXOpenAI
from core.model_providers.error import LLMBadRequestError, LLMAPIConnectionError, LLMAPIUnavailableError, \
    LLMRateLimitError, LLMAuthorizationError
from core.model_providers.models.llm.base import BaseLLM
from core.model_providers.models.entity.message import PromptMessage, MessageType
from core.model_providers.models.entity.model_params import ModelMode, ModelKwargs

CHAT_MODELS = [
    'gpt-3.5-turbo', # 4,096 tokens
    'yt-chat-v010', # 4096 tokens
    'yt-chat-v026', # 4096 tokens
    'pygmalion-2-7b', # 4096 tokens
    'Qwen-Chat', # 16384 tokens
]

MODEL_MAX_TOKENS = {
    'gpt-3.5-turbo': 4096,
    'yt-chat-v010': 4096,
    'yt-chat-v026': 4096,
    'pygmalion-2-7b': 4096,
    'Qwen-Chat': 16384,
}


class XOpenAIModel(BaseLLM):
    def __init__(self, model_provider: BaseModelProvider,
                 name: str,
                 model_kwargs: ModelKwargs,
                 streaming: bool = False,
                 callbacks: Callbacks = None):
        self.model_mode = ModelMode.CHAT
        
        # TODO load price config from configs(db)
        super().__init__(model_provider, name, model_kwargs, streaming, callbacks)

    def _init_client(self) -> Any:
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, self.model_kwargs)
        # Fine-tuning is currently only available for the following base models:
        # davinci, curie, babbage, and ada.
        # This means that except for the fixed `completion` model,
        # all other fine-tuned models are `completion` models.
        extra_model_kwargs = {
            'top_p': provider_model_kwargs.get('top_p'),
            'frequency_penalty': provider_model_kwargs.get('frequency_penalty'),
            'presence_penalty': provider_model_kwargs.get('presence_penalty'),
        }

        client = EnhanceChatXOpenAI(
            model_name=self.name,
            temperature=provider_model_kwargs.get('temperature'),
            max_tokens=provider_model_kwargs.get('max_tokens'),
            model_kwargs=extra_model_kwargs,
            streaming=self.streaming,
            callbacks=self.callbacks,
            request_timeout=60,
            **self.credentials
        )

        return client

    def _run(self, messages: List[PromptMessage],
             stop: Optional[List[str]] = None,
             callbacks: Callbacks = None,
             **kwargs) -> LLMResult:
        """
        run predict by prompt messages and stop words.

        :param messages:
        :param stop:
        :param callbacks:
        :return:
        """
        prompts = self._get_prompt_from_messages(messages)
        return self._client.generate([prompts], stop, callbacks)

    def get_num_tokens(self, messages: List[PromptMessage]) -> int:
        """
        get num tokens of prompt messages.

        :param messages:
        :return:
        """
        prompts = self._get_prompt_from_messages(messages)
        if isinstance(prompts, str):
            return self._client.get_num_tokens(prompts)
        else:
            return max(self._client.get_num_tokens_from_messages(prompts) - len(prompts), 0)

    def _set_model_kwargs(self, model_kwargs: ModelKwargs):
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, model_kwargs)
        extra_model_kwargs = {
            'top_p': provider_model_kwargs.get('top_p'),
            'frequency_penalty': provider_model_kwargs.get('frequency_penalty'),
            'presence_penalty': provider_model_kwargs.get('presence_penalty'),
        }

        self.client.temperature = provider_model_kwargs.get('temperature')
        self.client.max_tokens = provider_model_kwargs.get('max_tokens')
        self.client.model_kwargs = extra_model_kwargs

    def handle_exceptions(self, ex: Exception) -> Exception:
        traceback.print_exc()
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
            return LLMRateLimitError(str(ex))
        elif isinstance(ex, openai.error.AuthenticationError):
            return LLMAuthorizationError(str(ex))
        elif isinstance(ex, openai.error.OpenAIError):
            return LLMBadRequestError(ex.__class__.__name__ + ":" + str(ex))
        else:
            return ex

    @classmethod
    def support_streaming(cls):
        return True

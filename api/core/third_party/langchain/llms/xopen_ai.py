import json
import os
import requests

from typing import Dict, Any, Mapping, Optional, Union, Tuple
from langchain.llms.openai import OpenAI
from pydantic import root_validator


class EnhanceXOpenAI(OpenAI):
    request_timeout: Optional[Union[float, Tuple[float, float]]] = (5.0, 300.0)
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 1
    """Maximum number of retries to make when generating."""
    rest_api: str = "http://123.57.78.136/emb/api"
    """api to get infomations"""
    base_model_name: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            import openai

            values["client"] = openai.Completion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")
        return values

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**super()._invocation_params, **{
            "api_type": 'openai',
            "api_base": self.openai_api_base, # "http://123.57.78.136/emb/v1"
            "api_version": None,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization if self.openai_organization else None,
        }}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**super()._identifying_params, **{
            "api_type": 'openai',
            "api_base": self.openai_api_base, # "http://123.57.78.136/emb/v1"
            "api_version": None,
            "api_key": self.openai_api_key,
            "organization": self.openai_organization if self.openai_organization else None,
        }}

    def get_num_tokens(self, text: str) -> int:
        token_num_api = self.rest_api + '/v1/token_nums'
        payload = {
            "model": self.model_name,
            "max_tokens": 512
        }
        if text:
            payload["prompt"] = text
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(token_num_api, data=json.dumps(payload), headers=headers)

        return int(response.json()["tokenCount"])
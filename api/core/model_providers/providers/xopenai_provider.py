import json
import logging
from json import JSONDecodeError
from typing import Type, Optional

from flask import current_app
from openai.error import AuthenticationError, OpenAIError

import openai

from core.helper import encrypter
from core.model_providers.models.entity.provider import ModelFeature
from core.model_providers.models.base import BaseProviderModel
from core.model_providers.models.embedding.xopenai_embedding import XOpenAIEmbedding
from core.model_providers.models.entity.model_params import ModelKwargsRules, KwargRule, ModelType
from core.model_providers.models.llm.xopenai_model import XOpenAIModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from extensions.ext_database import db
from models.provider import ProviderType, ProviderModel


BASE_MODELS = [
    'gpt-3.5-turbo',
    'WizardLM-70B-V1.0',
    'yt-vicuna-13b',
    'Qwen-7B-Chat',
    'multilingual-e5-large',
    'bge-base-en',
]

class XOpenAIProvider(BaseModelProvider):

    @property
    def provider_name(self):
        """
        Returns the name of a provider.
        """
        return 'xopenai'

    def get_supported_model_list(self, model_type: ModelType) -> list[dict]:
        # convert old provider config to provider models
        self._convert_provider_config_to_model_config()

        if self.provider.provider_type == ProviderType.CUSTOM.value:
            # get configurable provider models
            provider_models = db.session.query(ProviderModel).filter(
                ProviderModel.tenant_id == self.provider.tenant_id,
                ProviderModel.provider_name == self.provider.provider_name,
                ProviderModel.model_type == model_type.value,
                ProviderModel.is_valid == True
            ).order_by(ProviderModel.created_at.asc()).all()

            model_list = []
            for provider_model in provider_models:
                model_dict = {
                    'id': provider_model.model_name,
                    'name': provider_model.model_name
                }

                credentials = json.loads(provider_model.encrypted_config)
                print(f"<<<<<<<<<<<<<<<<<<<<<{credentials}")
                if credentials['base_model_name'] in [
                    'gpt-4',
                    'gpt-4-32k',
                    'gpt-3.5-turbo',
                    'gpt-3.5-turbo-16k',
                ]:
                    model_dict['features'] = [
                        ModelFeature.AGENT_THOUGHT.value
                    ]

                model_list.append(model_dict)
        else:
            model_list = self._get_fixed_model_list(model_type)

        return model_list
    
    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if model_type == ModelType.TEXT_GENERATION:
            models = [
                {
                    'id': 'gpt-3.5-turbo',
                    'name': 'gpt-3.5-turbo',
                    'features': [
                        ModelFeature.AGENT_THOUGHT.value
                    ]
                },
                {
                    'id': 'WizardLM-70B-V1.0',
                    'name': 'WizardLM-70B-V1.0',
                    'features': [
                        ModelFeature.AGENT_THOUGHT.value
                    ]
                },
                {
                    'id': 'yt-vicuna-13b',
                    'name': 'yt-vicuna-13b',
                    'features': [
                        ModelFeature.AGENT_THOUGHT.value
                    ]
                },
                {
                    'id': 'Qwen-7B-Chat',
                    'name': 'Qwen-7B-Chat',
                    'features': [
                        ModelFeature.AGENT_THOUGHT.value
                    ]
                },
            ]

            return models
        elif model_type == ModelType.EMBEDDINGS:
            return [
                {
                    'id': 'multilingual-e5-large',
                    'name': 'multilingual-e5-large'
                },
                {
                    'id': 'bge-base-en',
                    'name': 'bge-base-en'
                },
            ]
        else:
            return []

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        """
        Returns the model class.

        :param model_type:
        :return:
        """
        if model_type == ModelType.TEXT_GENERATION:
            model_class = XOpenAIModel
        elif model_type == ModelType.EMBEDDINGS:
            model_class = XOpenAIEmbedding
        else:
            raise NotImplementedError

        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        """
        get model parameter rules.

        :param model_name:
        :param model_type:
        :return:
        """
        model_max_tokens = {
            'gpt-3.5-turbo': 4096,
            'WizardLM-70B-V1.0': 4096,
            'yt-vicuna-13b': 4096,
            'Qwen-7B-Chat': 16384,
        }
        model_credentials = self.get_model_credentials(model_name, model_type)
        return ModelKwargsRules(
            temperature=KwargRule[float](min=0, max=2, default=1),
            top_p=KwargRule[float](min=0, max=1, default=1),
            presence_penalty=KwargRule[float](min=-2, max=2, default=0),
            frequency_penalty=KwargRule[float](min=-2, max=2, default=0),
            max_tokens=KwargRule[int](min=10, max=model_max_tokens.get(
                model_credentials['base_model_name'],
                4097
            ), default=16),
        )

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        """
        Validates the given credentials.
        """
        if 'openai_api_base' not in credentials:
            raise CredentialsValidateFailedError('XOpenAI API Base Endpoint is required')
        
        if 'base_model_name' not in credentials:
            raise CredentialsValidateFailedError('Base Model Name is required')
        
        if credentials['base_model_name'] not in BASE_MODELS:
            raise CredentialsValidateFailedError('Base Model Name is invalid')
        
        try:
            credentials_kwargs = {"api_key": "EMPTY"}

            if credentials.get('openai_api_base'):
                credentials_kwargs['api_base'] = credentials['openai_api_base'] + '/v1'
        except (AuthenticationError, OpenAIError) as ex:
            raise CredentialsValidateFailedError(str(ex))
        
        if model_type == ModelType.TEXT_GENERATION:
            try:
                openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": 'ping'}],
                    model=model_name,
                    timeout=10,
                    request_timeout=(5, 30),
                    max_tokens=20,
                    **credentials_kwargs
                )
            except Exception as ex:
                logging.exception('XOpenAI config validation failed')
                raise ex
        elif model_type == ModelType.EMBEDDINGS:
            try:
                openai.Embedding.create(
                    model=model_name, 
                    input=["ping"], 
                    timeout=10,
                    request_timeout=(5, 30),
                    **credentials_kwargs)
            except Exception as e:
                logging.exception("XOpenAI Model retrieve failed.")
                raise e
            
    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType,
                                  credentials: dict) -> dict:
        credentials['openai_api_base'] = encrypter.encrypt_token(tenant_id, credentials['openai_api_base'])
        return credentials

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool = False) -> dict:
        """
        get credentials for llm use.

        :param model_name:
        :param model_type:
        :param obfuscated:
        :return:
        """
        if self.provider.provider_type == ProviderType.CUSTOM.value:
            # convert old provider config to provider models
            self._convert_provider_config_to_model_config()

            provider_model = self._get_provider_model(model_name, model_type)

            if not provider_model.encrypted_config:
                return {
                    'openai_api_base': '',
                    'openai_api_key': '',
                    'base_model_name': ''
                }

            credentials = json.loads(provider_model.encrypted_config)
            if not credentials.get('openai_api_base'):
                credentials['openai_api_base'] = None
            else:
                credentials['openai_api_base'] = credentials['openai_api_base'] + '/v1'

            return credentials
        else:
            return {
                'openai_api_base': None,
                'openai_api_key': None,
                'base_model_name': None
            }
            
    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        """
        check model credentials valid.

        :param model_name:
        :param model_type:
        :param credentials:
        """
        return

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType, credentials: dict) -> dict:
        """
        encrypt model credentials for save.

        :param tenant_id:
        :param model_name:
        :param model_type:
        :param credentials:
        :return:
        """
        return {}

    def get_provider_credentials(self, obfuscated: bool = False) -> dict:
        return {}
    
    def _convert_provider_config_to_model_config(self):
        if self.provider.provider_type == ProviderType.CUSTOM.value \
                and self.provider.is_valid \
                and self.provider.encrypted_config:
            try:
                credentials = json.loads(self.provider.encrypted_config)
            except JSONDecodeError:
                credentials = {
                    'openai_api_base': '',
                    'openai_api_key': '',
                    'base_model_name': ''
                }

            self._add_provider_model(
                model_name='gpt-3.5-turbo',
                model_type=ModelType.TEXT_GENERATION,
                provider_credentials=credentials
            )

            self._add_provider_model(
                model_name='WizardLM-70B-V1.0',
                model_type=ModelType.TEXT_GENERATION,
                provider_credentials=credentials
            )

            self._add_provider_model(
                model_name='yt-vicuna-13b',
                model_type=ModelType.TEXT_GENERATION,
                provider_credentials=credentials
            )

            self._add_provider_model(
                model_name='Qwen-7B-Chat',
                model_type=ModelType.TEXT_GENERATION,
                provider_credentials=credentials
            )

            self._add_provider_model(
                model_name='multilingual-e5-large',
                model_type=ModelType.EMBEDDINGS,
                provider_credentials=credentials
            )

            self._add_provider_model(
                model_name='bge-base-en',
                model_type=ModelType.EMBEDDINGS,
                provider_credentials=credentials
            )

            self.provider.encrypted_config = None
            db.session.commit()

    def _add_provider_model(self, model_name: str, model_type: ModelType, provider_credentials: dict):
        credentials = provider_credentials.copy()
        credentials['base_model_name'] = model_name
        provider_model = ProviderModel(
            tenant_id=self.provider.tenant_id,
            provider_name=self.provider.provider_name,
            model_name=model_name,
            model_type=model_type.value,
            encrypted_config=json.dumps(credentials),
            is_valid=True
        )
        db.session.add(provider_model)
        db.session.commit()
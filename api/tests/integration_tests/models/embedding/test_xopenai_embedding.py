import json
import os
from unittest.mock import patch, MagicMock

from core.model_providers.models.embedding.xopenai_embedding import XOpenAIEmbedding
from core.model_providers.models.entity.model_params import ModelType
from core.model_providers.providers.xopenai_provider import XOpenAIProvider
from models.provider import Provider, ProviderType, ProviderModel


def get_mock_provider():
    return Provider(
        id='provider_id',
        tenant_id='tenant_id',
        provider_name='xopenai',
        provider_type=ProviderType.CUSTOM.value,
        encrypted_config='',
        is_valid=True,
    )


def get_mock_embedding_model(mocker):
    model_name = 'multilingual-e5-large'
    openai_api_base = 'http://124.71.148.73/emb'
    openai_api_key = 'EMPTY'
    base_model_name = 'multilingual-e5-large'
    model_provider = XOpenAIProvider(provider=get_mock_provider())

    mock_query = MagicMock()
    mock_query.filter.return_value.first.return_value = ProviderModel(
        provider_name='xopenai',
        model_name=model_name,
        model_type=ModelType.EMBEDDINGS.value,
        encrypted_config=json.dumps({
            'openai_api_base': openai_api_base,
            'openai_api_key': openai_api_key,
            'base_model_name': base_model_name
        }),
        is_valid=True,
    )
    mocker.patch('extensions.ext_database.db.session.query', return_value=mock_query)

    return XOpenAIEmbedding(
        model_provider=model_provider,
        name=model_name
    )


def decrypt_side_effect(tenant_id, encrypted_openai_api_base):
    return encrypted_openai_api_base


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_embed_documents(mock_decrypt, mocker):
    embedding_model = get_mock_embedding_model(mocker)
    rst = embedding_model.client.embed_documents(['test', 'test1'])
    assert isinstance(rst, list)
    assert len(rst) == 2
    assert len(rst[0]) == 1024


@patch('core.helper.encrypter.decrypt_token', side_effect=decrypt_side_effect)
def test_embed_query(mock_decrypt, mocker):
    embedding_model = get_mock_embedding_model(mocker)
    rst = embedding_model.client.embed_query('test')
    assert isinstance(rst, list)
    assert len(rst) == 1024

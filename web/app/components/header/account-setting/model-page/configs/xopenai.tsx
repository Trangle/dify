import { ProviderEnum } from '../declarations'
import type { ProviderConfig } from '../declarations'
import { OpenaiGreen, OpenaiText, OpenaiTransparent } from '@/app/components/base/icons/src/public/llm'

const config: ProviderConfig = {
  selector: {
    name: {
      'en': 'XOpenAI',
      'zh-Hans': 'XOpenAI',
    },
    icon: <OpenaiGreen className='w-full h-full' />,
  },
  item: {
    key: ProviderEnum.xopenai,
    titleIcon: {
      'en': <OpenaiText className='h-5' />,
      'zh-Hans': <OpenaiText className='h-5' />,
    },
    subTitleIcon: <OpenaiGreen className='w-6 h-6' />,
    desc: {
      'en': 'Models provided by XOpenAI Compatible API, such as vicuna-v1.5 and Qwen-XB-Chat.',
      'zh-Hans': 'XOpenAI 提供的模型，例如 vicuna-v1.5 和 Qwen-XB-Chat。',
    },
    bgColor: 'bg-gray-200',
  },
  modal: {
    key: ProviderEnum.xopenai,
    title: {
      'en': 'XOpenAI',
      'zh-Hans': 'XOpenAI',
    },
    icon: <OpenaiTransparent className='w-6 h-6' />,
    link: {
      href: 'https://github.com/lm-sys/FastChat',
      label: {
        'en': 'How to deploy XOpenAI',
        'zh-Hans': '如何部署 XOpenAI',
      },
    },
    defaultValue: {
      model_type: 'text-generation',
    },
    validateKeys: [
      'model_name',
      'model_type',
      'openai_api_base',
      'base_model_name',
    ],
    fields: [
      {
        type: 'text',
        key: 'model_name',
        required: true,
        label: {
          'en': 'Deployment Name',
          'zh-Hans': '部署名称',
        },
        placeholder: {
          'en': 'Enter your Deployment Name here',
          'zh-Hans': '在此输入您的部署名称',
        },
      },
      {
        type: 'radio',
        key: 'model_type',
        required: true,
        label: {
          'en': 'Model Type',
          'zh-Hans': '模型类型',
        },
        options: [
          {
            key: 'text-generation',
            label: {
              'en': 'Text Generation',
              'zh-Hans': '文本生成',
            },
          },
          {
            key: 'embeddings',
            label: {
              'en': 'Embeddings',
              'zh-Hans': 'Embeddings',
            },
          },
        ],
      },
      {
        type: 'text',
        key: 'openai_api_base',
        required: true,
        label: {
          'en': 'API Endpoint URL',
          'zh-Hans': 'API 域名',
        },
        placeholder: {
          'en': 'Enter your API Endpoint, eg: https://example.com/xxx',
          'zh-Hans': '在此输入您的 API 域名，如：https://example.com/xxx',
        },
      },
      {
        type: 'select',
        key: 'base_model_name',
        required: true,
        label: {
          'en': 'Base Model',
          'zh-Hans': '基础模型',
        },
        options: (v) => {
          if (v.model_type === 'text-generation') {
            return [
              {
                key: 'gpt-3.5-turbo',
                label: {
                  'en': 'gpt-3.5-turbo',
                  'zh-Hans': 'gpt-3.5-turbo',
                },
              },
              {
                key: 'yt-chat-v010',
                label: {
                  'en': 'yt-chat-v010',
                  'zh-Hans': 'yt-chat-v010',
                },
              },
              {
                key: 'pygmalion-2-7b',
                label: {
                  'en': 'pygmalion-2-7b',
                  'zh-Hans': 'pygmalion-2-7b',
                },
              },
              {
                key: 'Qwen-Chat',
                label: {
                  'en': 'Qwen-Chat',
                  'zh-Hans': 'Qwen-Chat',
                },
              },
            ]
          }
          if (v.model_type === 'embeddings') {
            return [
              {
                key: 'multilingual-e5-large',
                label: {
                  'en': 'multilingual-e5-large',
                  'zh-Hans': 'multilingual-e5-large',
                },
              },
              {
                key: 'bge-base-en',
                label: {
                  'en': 'bge-base-en',
                  'zh-Hans': 'bge-base-en',
                },
              },
            ]
          }
          return []
        },
      },
    ],
  },
}
export default config

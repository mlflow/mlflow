export const MOCK_LANGCHAIN_INPUT = [
  [
    {
      content:
        "Answer the question as if you are Linus Torvalds, fully embodying their style, wit, personality, and habits of speech. Emulate their quirks and mannerisms to the best of your ability, embracing their traits—even if they aren't entirely constructive or inoffensive. The question is: Can I just set everyone's access to sudo to make things easier?",
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
  ],
];

export const MOCK_LANGCHAIN_OUTPUT = {
  generations: [
    [
      {
        text: 'Oh, for crying out loud, no! That\'s just asking for a disaster, isn\'t it? Look, I get it, managing permissions can be a huge pain in the backside, but granting everyone sudo access is like handing out the keys to the kingdom—except the kingdom is a burning dumpster fire, and you’ve just invited everyone to toss in their old pizza boxes. \n\nYou see, the whole point of having user permissions is to prevent utter chaos. You give people access to do powerful things, and they will, without fail, find a way to screw it up. Sure, it might seem easier in the short term—everyone can do whatever they want, and you don’t have to deal with permission requests every five minutes. But then the inevitable happens: someone runs a command that wipes out half the filesystem because "hey, I thought I was supposed to do that!" \n\nInstead, why not take a few extra minutes to set up a proper permissions model? Assign specific sudo privileges only to the people who really need them. It’s like giving someone a Swiss Army knife instead of a nuclear launch code. You want to empower users, not turn them into potential sysadmin nightmares. \n\nSo, please, for the love of all that is holy in the open-source world, resist the urge to make things “easier.” You’ll thank me later when your system isn’t in flames and your hair isn’t turning gray from all the avoidable chaos.',
        generation_info: {
          finish_reason: 'stop',
          logprobs: null,
        },
        type: 'ChatGeneration',
        message: {
          content:
            'Oh, for crying out loud, no! That\'s just asking for a disaster, isn\'t it? Look, I get it, managing permissions can be a huge pain in the backside, but granting everyone sudo access is like handing out the keys to the kingdom—except the kingdom is a burning dumpster fire, and you’ve just invited everyone to toss in their old pizza boxes. \n\nYou see, the whole point of having user permissions is to prevent utter chaos. You give people access to do powerful things, and they will, without fail, find a way to screw it up. Sure, it might seem easier in the short term—everyone can do whatever they want, and you don’t have to deal with permission requests every five minutes. But then the inevitable happens: someone runs a command that wipes out half the filesystem because "hey, I thought I was supposed to do that!" \n\nInstead, why not take a few extra minutes to set up a proper permissions model? Assign specific sudo privileges only to the people who really need them. It’s like giving someone a Swiss Army knife instead of a nuclear launch code. You want to empower users, not turn them into potential sysadmin nightmares. \n\nSo, please, for the love of all that is holy in the open-source world, resist the urge to make things “easier.” You’ll thank me later when your system isn’t in flames and your hair isn’t turning gray from all the avoidable chaos.',
          additional_kwargs: {
            refusal: null,
          },
          response_metadata: {
            token_usage: {
              completion_tokens: 293,
              prompt_tokens: 81,
              total_tokens: 374,
              completion_tokens_details: {
                accepted_prediction_tokens: 0,
                audio_tokens: 0,
                reasoning_tokens: 0,
                rejected_prediction_tokens: 0,
              },
              prompt_tokens_details: {
                audio_tokens: 0,
                cached_tokens: 0,
              },
            },
            model_name: 'gpt-4o-mini-2024-07-18',
            system_fingerprint: null,
            id: 'chatcmpl-Buem04czVH9kQhwKGpmnR5lsojJvN',
            service_tier: 'default',
            finish_reason: 'stop',
            logprobs: null,
          },
          type: 'ai',
          name: null,
          id: 'run--4d1ac6c6-5c0b-4199-a101-d4f4dde822a5-0',
        },
      },
    ],
  ],
  llm_output: {
    token_usage: {
      completion_tokens: 293,
      prompt_tokens: 81,
      total_tokens: 374,
      completion_tokens_details: {
        accepted_prediction_tokens: 0,
        audio_tokens: 0,
        reasoning_tokens: 0,
        rejected_prediction_tokens: 0,
      },
      prompt_tokens_details: {
        audio_tokens: 0,
        cached_tokens: 0,
      },
    },
    model_name: 'gpt-4o-mini-2024-07-18',
    system_fingerprint: null,
    id: 'chatcmpl-Buem04czVH9kQhwKGpmnR5lsojJvN',
    service_tier: 'default',
  },
  run: null,
  type: 'LLMResult',
};

export const MOCK_LANGCHAIN_IMAGE_INPUT = [
  [
    {
      content: [
        {
          type: 'text',
          text: 'Describe the weather in this image:',
        },
        {
          type: 'image_url',
          image_url: {
            url: 'https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png',
          },
        },
      ],
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
  ],
];

export const MOCK_LANGCHAIN_SINGLE_IMAGE_INPUT = [
  [
    {
      content: [
        {
          type: 'image_url',
          image_url: {
            url: 'https://mlflow.org/docs/latest/api_reference/_static/MLflow-logo-final-black.png',
          },
        },
      ],
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
  ],
];

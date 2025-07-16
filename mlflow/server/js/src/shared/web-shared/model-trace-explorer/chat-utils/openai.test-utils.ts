export const MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT = [
  {
    type: 'response.output_item.done',
    custom_outputs: null,
    item: {
      type: 'function_call',
      call_id: 'call_fltScw1rzGLyDEAWY2rEev3U',
      name: 'telco_customer_support_dev__agent__get_billing_info',
      arguments: '{"billing_start_date": "2024-10-01", "billing_end_date": "2024-10-31"}',
      id: '02ce3a91-fcc7-425e-9038-65118023b9f2',
    },
  },
  {
    type: 'response.output_item.done',
    custom_outputs: null,
    item: {
      type: 'function_call_output',
      call_id: 'call_fltScw1rzGLyDEAWY2rEev3U',
      output: '[{"customer_id":"CUS-10045","billing_id":"BILL-1234568792"}]',
    },
  },
  {
    type: 'response.output_item.done',
    custom_outputs: null,
    item: {
      role: 'assistant',
      type: 'message',
      id: 'd976f842-3a9b-4524-9822-c9860e9a483a',
      content: [
        {
          type: 'output_text',
          text: 'It seems I was able to retrieve your billing details',
        },
      ],
    },
  },
];

export const MOCK_OPENAI_RESPONSES_OUTPUT = [
  {
    type: 'function_call',
    call_id: 'call_UhQLOPvX1OFzM4vqN7JaB9Xo',
    name: 'support_tickets_vector_search',
    arguments: '{"query": "iPhone won\'t connect to international networks", "filters": null}',
    id: 'e7cb4d8a-1322-4990-87dd-ae590fa9534b',
  },
  {
    type: 'function_call_output',
    call_id: 'call_UhQLOPvX1OFzM4vqN7JaB9Xo',
    output: "[{'page_content': 'Content'}]",
  },
  {
    type: 'message',
    role: 'assistant',
    id: '51b35734-4c24-40ae-88f1-3ff73c18a2c1',
    content: [
      {
        type: 'output_text',
        text: 'Test output text',
      },
    ],
  },
  {
    id: 'msg_683fbc587468819a82bff3d65e84a4f307d127167631bb25',
    content: [
      {
        annotations: [],
        text: 'Not much! Just here and ready to help you out',
        type: 'output_text',
        logprobs: null,
      },
    ],
    role: 'assistant',
    status: 'completed',
    type: 'message',
  },
];

export const MOCK_OPENAI_RESPONSES_INPUT = {
  input: [
    {
      status: null,
      content: 'Test content',
      role: 'user',
      type: 'message',
    },
  ],
};

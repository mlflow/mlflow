import { describe, it, expect } from '@jest/globals';
import type { RegisteredPromptVersion } from '../../prompts/types';
import { buildMessagesFromVersion } from './PromptRegistryPicker';

const buildVersion = (overrides: Partial<RegisteredPromptVersion>): RegisteredPromptVersion =>
  ({
    name: 'p',
    version: '1',
    creation_timestamp: 0,
    last_updated_timestamp: 0,
    current_stage: 'None',
    description: '',
    source: 'dummy',
    run_id: '',
    status: 'READY',
    user_id: '',
    tags: [],
    ...overrides,
  }) as RegisteredPromptVersion;

describe('buildMessagesFromVersion', () => {
  it('parses a chat-typed prompt into its messages, preserving roles', () => {
    const messages = [
      { role: 'system', content: 'You are a concise summarizer.' },
      { role: 'user', content: 'Summarize: {{ text }}' },
    ];
    const version = buildVersion({
      tags: [
        { key: '_mlflow_prompt_type', value: 'chat' },
        { key: 'mlflow.prompt.text', value: JSON.stringify(messages) },
      ],
    });

    expect(buildMessagesFromVersion(version)).toEqual(messages);
  });

  it('coerces unknown chat roles to user', () => {
    const version = buildVersion({
      tags: [
        { key: '_mlflow_prompt_type', value: 'chat' },
        {
          key: 'mlflow.prompt.text',
          value: JSON.stringify([
            { role: 'tool', content: 'tool output' },
            { role: 'user', content: 'follow up' },
          ]),
        },
      ],
    });

    expect(buildMessagesFromVersion(version)).toEqual([
      { role: 'user', content: 'tool output' },
      { role: 'user', content: 'follow up' },
    ]);
  });

  it('wraps a text-typed prompt as a single user message', () => {
    const version = buildVersion({
      tags: [
        { key: '_mlflow_prompt_type', value: 'text' },
        { key: 'mlflow.prompt.text', value: 'Translate to French: {{ text }}' },
      ],
    });

    expect(buildMessagesFromVersion(version)).toEqual([{ role: 'user', content: 'Translate to French: {{ text }}' }]);
  });

  it('returns an empty list when a chat-typed prompt has unparseable content', () => {
    const version = buildVersion({
      tags: [
        { key: '_mlflow_prompt_type', value: 'chat' },
        { key: 'mlflow.prompt.text', value: '{not json}' },
      ],
    });

    expect(buildMessagesFromVersion(version)).toEqual([]);
  });
});

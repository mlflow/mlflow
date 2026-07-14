import { describe, it, expect } from '@jest/globals';
import type { ChatMessage, PlaygroundParams } from './types';
import { getSaveableMessages, hasSaveableSettings, paramsToModelConfig } from './promptVersionSave';

describe('paramsToModelConfig', () => {
  it('returns undefined when no params are set', () => {
    expect(paramsToModelConfig({})).toBeUndefined();
  });

  it('maps set params into the registry snake_case shape', () => {
    const params: PlaygroundParams = {
      temperature: 0.7,
      max_tokens: 256,
      top_p: 0.9,
      top_k: 40,
      frequency_penalty: 0.1,
      presence_penalty: -0.2,
    };
    expect(paramsToModelConfig(params)).toEqual({
      temperature: 0.7,
      max_tokens: 256,
      top_p: 0.9,
      top_k: 40,
      frequency_penalty: 0.1,
      presence_penalty: -0.2,
    });
  });

  it('renames stop to stop_sequences and drops an empty list', () => {
    expect(paramsToModelConfig({ stop: ['END', 'STOP'] })).toEqual({ stop_sequences: ['END', 'STOP'] });
    expect(paramsToModelConfig({ stop: [] })).toBeUndefined();
  });

  it('keeps a zero value (does not treat 0 as unset)', () => {
    expect(paramsToModelConfig({ temperature: 0 })).toEqual({ temperature: 0 });
  });
});

describe('getSaveableMessages', () => {
  it('trims content and drops empty messages', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: '  Be concise.  ' },
      { role: 'user', content: 'Summarize {{ text }}' },
      { role: 'user', content: '   ' },
      { role: 'user', content: '' },
    ];
    expect(getSaveableMessages(messages)).toEqual([
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Summarize {{ text }}' },
    ]);
  });

  it('drops assistant (model-generated) turns so only the template is saved', () => {
    const messages: ChatMessage[] = [
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello! How can I help?' },
      { role: 'user', content: 'Summarize {{ text }}' },
    ];
    expect(getSaveableMessages(messages)).toEqual([
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Hi' },
      { role: 'user', content: 'Summarize {{ text }}' },
    ]);
  });

  it('returns an empty list when every message is blank', () => {
    expect(getSaveableMessages([{ role: 'user', content: '   ' }])).toEqual([]);
  });
});

describe('hasSaveableSettings', () => {
  it('is true when model config is present', () => {
    expect(hasSaveableSettings({ temperature: 0.5 }, 'text', '')).toBe(true);
  });

  it('is true when a json_schema response format is present', () => {
    expect(hasSaveableSettings({}, 'json_schema', '{"type":"object"}')).toBe(true);
  });

  it('is false when there is no config and no schema', () => {
    expect(hasSaveableSettings({}, 'text', '')).toBe(false);
    expect(hasSaveableSettings({}, 'json_schema', '   ')).toBe(false);
  });
});

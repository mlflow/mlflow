import { describe, it, expect } from '@jest/globals';
import { buildPlaygroundDatasetRecord, getDatasetInputMessages, getLatestAssistantContent } from './datasetRecord';
import type { ConversationMessage } from './types';

describe('getLatestAssistantContent', () => {
  it('returns the content of the most recent assistant reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'First' },
      { role: 'user', content: 'Again' },
      { role: 'assistant', content: 'Second' },
      { role: 'user', content: '' },
    ];
    expect(getLatestAssistantContent(messages)).toBe('Second');
  });

  it('returns an empty string when there is no assistant reply yet', () => {
    expect(getLatestAssistantContent([{ role: 'user', content: 'Hi' }])).toBe('');
  });

  it('returns an empty string once the user has typed a new turn after the last assistant reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
      { role: 'user', content: 'A brand new question' },
    ];
    expect(getLatestAssistantContent(messages)).toBe('');
  });

  it('treats a null assistant content (e.g. a tool-call-only reply) as an empty string', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: null },
    ];
    expect(getLatestAssistantContent(messages)).toBe('');
  });
});

describe('getDatasetInputMessages', () => {
  it('resolves variables and drops the assistant reply and the trailing empty composer', () => {
    const messages: ConversationMessage[] = [
      { role: 'system', content: 'You are {{ persona }}.' },
      { role: 'user', content: 'Summarize {{ topic }}' },
      { role: 'assistant', content: 'Here is the summary', usage: { total_tokens: 10 } },
      { role: 'user', content: '' },
    ];
    expect(getDatasetInputMessages(messages, { persona: 'concise', topic: 'MLflow' })).toEqual([
      { role: 'system', content: 'You are concise.' },
      { role: 'user', content: 'Summarize MLflow' },
    ]);
  });

  it('keeps prior turns (including earlier assistant replies) as multi-turn input context', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
      { role: 'user', content: 'Tell me more' },
      { role: 'assistant', content: 'Sure, here it is' },
      { role: 'user', content: '' },
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
      { role: 'user', content: 'Tell me more' },
    ]);
  });

  it('keeps the whole conversation, reply included, once a new non-empty turn follows the last assistant reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello', usage: { total_tokens: 5 } },
      { role: 'user', content: 'A brand new question' },
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
      { role: 'user', content: 'A brand new question' },
    ]);
  });

  it('returns every non-empty turn when the prompt has not been run yet', () => {
    const messages: ConversationMessage[] = [
      { role: 'system', content: 'Be helpful' },
      { role: 'user', content: 'Hello' },
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual([
      { role: 'system', content: 'Be helpful' },
      { role: 'user', content: 'Hello' },
    ]);
  });

  it('filters out empty-content turns', () => {
    const messages: ConversationMessage[] = [
      { role: 'system', content: '   ' },
      { role: 'user', content: 'Only this' },
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual([{ role: 'user', content: 'Only this' }]);
  });
});

describe('buildPlaygroundDatasetRecord', () => {
  it('stores inputs as { messages } and a non-empty expected response under expectations', () => {
    const record = buildPlaygroundDatasetRecord({
      inputMessages: [{ role: 'user', content: 'Question' }],
      expectedResponse: 'Answer',
    });
    expect(record).toEqual({
      inputs: { messages: [{ role: 'user', content: 'Question' }] },
      expectations: { expected_response: 'Answer' },
    });
  });

  it('trims the expected response before storing it', () => {
    const record = buildPlaygroundDatasetRecord({
      inputMessages: [{ role: 'user', content: 'Question' }],
      expectedResponse: '  Answer  ',
    });
    expect(record.expectations).toEqual({ expected_response: 'Answer' });
  });

  it('omits expectations entirely when the expected response is blank', () => {
    const record = buildPlaygroundDatasetRecord({
      inputMessages: [{ role: 'user', content: 'Question' }],
      expectedResponse: '   ',
    });
    expect(record).toEqual({ inputs: { messages: [{ role: 'user', content: 'Question' }] } });
    expect(record).not.toHaveProperty('expectations');
  });
});

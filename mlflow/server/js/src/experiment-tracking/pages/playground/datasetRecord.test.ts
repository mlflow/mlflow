import { describe, it, expect } from '@jest/globals';
import { buildPlaygroundDatasetRecord, getDatasetInputMessages, getInputTurns, getLatestReply } from './datasetRecord';
import type { ChatMessage, ConversationMessage } from './types';

// A reply as the playground appends it after a run: tagged with the request that produced it.
const generatedReply = (
  content: string | null,
  requestMessages: ChatMessage[],
  extra: Partial<ConversationMessage> = {},
): ConversationMessage => ({ role: 'assistant', content, requestMessages, ...extra });

const EMPTY_COMPOSER: ConversationMessage = { role: 'user', content: '' };

describe('getDatasetInputMessages', () => {
  it('resolves variables and drops the generated reply and the trailing empty composer', () => {
    const messages: ConversationMessage[] = [
      { role: 'system', content: 'You are {{ persona }}.' },
      { role: 'user', content: 'Summarize {{ topic }}' },
      generatedReply(
        'Here is the summary',
        [
          { role: 'system', content: 'You are concise.' },
          { role: 'user', content: 'Summarize MLflow' },
        ],
        { usage: { total_tokens: 10 } },
      ),
      EMPTY_COMPOSER,
    ];
    expect(getDatasetInputMessages(messages, { persona: 'concise', topic: 'MLflow' })).toEqual([
      { role: 'system', content: 'You are concise.' },
      { role: 'user', content: 'Summarize MLflow' },
    ]);
  });

  it('keeps prior turns (including earlier replies) as multi-turn input context', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      generatedReply('Hello', [{ role: 'user', content: 'Hi' }]),
      { role: 'user', content: 'Tell me more' },
      generatedReply('Sure, here it is', [
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'Tell me more' },
      ]),
      EMPTY_COMPOSER,
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
      { role: 'user', content: 'Tell me more' },
    ]);
  });

  it('keeps the whole conversation, reply included, once a new non-empty turn follows the reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      generatedReply('Hello', [{ role: 'user', content: 'Hi' }]),
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

  it('keeps a trailing assistant turn the playground did not generate, such as a registry prefill', () => {
    const messages: ConversationMessage[] = [
      { role: 'system', content: 'Translate to French.' },
      { role: 'user', content: 'Good morning' },
      { role: 'assistant', content: 'Bien sûr ! Voici la traduction :' },
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual(messages);
  });

  it('keeps an assistant turn that only called a tool, with its tool calls', () => {
    const toolCalls = [{ id: 'call-1', type: 'function' as const, function: { name: 'get_weather', arguments: '{}' } }];
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Weather in Paris?' },
      generatedReply('', [{ role: 'user', content: 'Weather in Paris?' }], { tool_calls: toolCalls }),
      { role: 'user', content: 'And in Tokyo?' },
    ];
    expect(getDatasetInputMessages(messages, {})).toEqual([
      { role: 'user', content: 'Weather in Paris?' },
      { role: 'assistant', content: '', tool_calls: toolCalls },
      { role: 'user', content: 'And in Tokyo?' },
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

describe('getInputTurns', () => {
  it('returns the unresolved turns before the generated reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Summarize {{ topic }} for {{ audience }}' },
      generatedReply('Summary', [{ role: 'user', content: 'Summarize MLflow for engineers' }]),
      EMPTY_COMPOSER,
    ];
    expect(getInputTurns(messages)).toEqual([messages[0]]);
  });
});

describe('getLatestReply', () => {
  const question: ConversationMessage = { role: 'user', content: 'What is the capital of {{ country }}?' };
  const runWithFrance: ConversationMessage[] = [
    question,
    generatedReply('Paris', [{ role: 'user', content: 'What is the capital of France?' }]),
    EMPTY_COMPOSER,
  ];

  it('returns the reply that answers the current inputs', () => {
    expect(getLatestReply(runWithFrance, { country: 'France' })).toEqual({ content: 'Paris', isOutdated: false });
  });

  it('withholds the reply when a variable changed after the run', () => {
    expect(getLatestReply(runWithFrance, { country: 'Japan' })).toEqual({ content: '', isOutdated: true });
  });

  it('withholds the reply when an input message was edited after the run', () => {
    const edited: ConversationMessage[] = [
      { role: 'user', content: 'What is the capital of Japan?' },
      ...runWithFrance.slice(1),
    ];
    expect(getLatestReply(edited, {})).toEqual({ content: '', isOutdated: true });
  });

  it('returns nothing when the prompt has not been run yet', () => {
    expect(getLatestReply([{ role: 'user', content: 'Hi' }], {})).toEqual({ content: '', isOutdated: false });
  });

  it('returns nothing once the user has typed a new turn after the reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Hi' },
      generatedReply('Hello', [{ role: 'user', content: 'Hi' }]),
      { role: 'user', content: 'A brand new question' },
    ];
    expect(getLatestReply(messages, {})).toEqual({ content: '', isOutdated: false });
  });

  it('does not treat an assistant turn loaded from the registry as the reply', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Good morning' },
      { role: 'assistant', content: 'Bien sûr ! Voici la traduction :' },
    ];
    expect(getLatestReply(messages, {})).toEqual({ content: '', isOutdated: false });
  });

  it('returns nothing for a reply that only called a tool', () => {
    const messages: ConversationMessage[] = [
      { role: 'user', content: 'Weather?' },
      generatedReply(null, [{ role: 'user', content: 'Weather?' }]),
    ];
    expect(getLatestReply(messages, {})).toEqual({ content: '', isOutdated: false });
  });
});

describe('buildPlaygroundDatasetRecord', () => {
  it('stores inputs as { messages }, a non-empty expected response, and a human source', () => {
    const record = buildPlaygroundDatasetRecord({
      inputMessages: [{ role: 'user', content: 'Question' }],
      expectedResponse: 'Answer',
    });
    expect(record).toEqual({
      inputs: { messages: [{ role: 'user', content: 'Question' }] },
      expectations: { expected_response: 'Answer' },
      source: { source_type: 'HUMAN', source_data: {} },
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
    expect(record).not.toHaveProperty('expectations');
  });

  it('links the loaded registry prompt version through the mlflow.linkedPrompts tag', () => {
    const record = buildPlaygroundDatasetRecord({
      inputMessages: [{ role: 'user', content: 'Question' }],
      expectedResponse: '',
      linkedPrompt: { name: 'qa-prompt', version: '3' },
    });
    expect(record.tags).toEqual({ 'mlflow.linkedPrompts': JSON.stringify([{ name: 'qa-prompt', version: '3' }]) });
  });

  it('adds no tags when no registry prompt is loaded', () => {
    const record = buildPlaygroundDatasetRecord({
      inputMessages: [{ role: 'user', content: 'Question' }],
      expectedResponse: '',
      linkedPrompt: null,
    });
    expect(record).not.toHaveProperty('tags');
  });
});

import { describe, expect, test } from '@jest/globals';

import { typeSafeSystemOneTranslator } from './TypeSafeSystemOneTranslator';

const translate = ({ inputs, outputs }: { inputs: unknown; outputs: unknown }) =>
  typeSafeSystemOneTranslator.translate({ title: 'typesafe.system_one', inputs, outputs });

describe('TypeSafe System One source matching', () => {
  test.each([
    { title: 'typesafe.system_one' },
    { title: 'custom-name', modelProvider: 'typesafe' },
    { title: 'custom-name', modelProvider: 'TypeSafe' },
  ])('accepts a positive TypeSafe source signal: %p', (span) => {
    expect(typeSafeSystemOneTranslator.matches(span)).toBe(true);
  });

  test.each([{ title: 'typesafe.systemOne' }, { title: 'other', modelProvider: 'other' }, { title: 'jev-1.13.0' }])(
    'rejects spans without a TypeSafe source signal: %p',
    (span) => {
      expect(typeSafeSystemOneTranslator.matches(span)).toBe(false);
    },
  );

  test.each([
    { title: 'other', modelProvider: null },
    { title: 'other', modelProvider: 42 },
    { title: 'other', modelProvider: false },
    { title: 'other', modelProvider: ['typesafe'] },
    { title: 'other', modelProvider: { provider: 'typesafe' } },
  ])('rejects a non-string model provider without throwing: %p', (span) => {
    expect(typeSafeSystemOneTranslator.matches(span)).toBe(false);
  });
});

describe('TypeSafe System One decision translation', () => {
  test('parses question configuration and all supported answers without flattening JSON', () => {
    const inputs = {
      state: { message: 'Please fix my duplicate charge today.' },
      model: 'jev-latest',
      questions: {
        billing: {
          type: 'noul',
          instructions: { task: 'Is this about billing?' },
          criteria: { true: ['A charge is mentioned'], false: null },
        },
        tone: {
          type: 'choice',
          instructions: 'What is the tone?',
          criteria: { calm: null, urgent: { description: 'Needs prompt attention' } },
        },
        urgency: {
          type: 'score',
          instructions: ['Rate urgency'],
          criteria: ['Can wait', { label: 'This week' }, ['Today']],
        },
      },
    };
    const outputs = {
      model: 'jev-2026-09-15',
      usage: { input_tokens: 42, output_tokens: 7 },
      answers: {
        billing: { type: 'noul', noul: 0.98 },
        tone: {
          type: 'choice',
          choice: 'urgent',
          confidence: 0.84,
          probabilities: { calm: 0.16, urgent: 0.84 },
        },
        urgency: {
          type: 'score',
          score: 1.7,
          confidence: 0.72,
          legend: { '0': 'Can wait', '1': { label: 'This week' }, '2': ['Today'] },
          probabilities: { '0': 0.05, '1': 0.2, '2': 0.75 },
        },
      },
    };

    expect(translate({ inputs, outputs })).toEqual({
      inputs: {
        fields: [
          { kind: 'field', fieldKey: 'state' },
          {
            kind: 'questions',
            fieldKey: 'questions',
            items: [
              {
                id: 'billing',
                declaredType: 'noul',
                instructions: inputs.questions.billing.instructions,
                criteria: inputs.questions.billing.criteria,
                rawQuestion: inputs.questions.billing,
              },
              {
                id: 'tone',
                declaredType: 'choice',
                instructions: inputs.questions.tone.instructions,
                criteria: inputs.questions.tone.criteria,
                rawQuestion: inputs.questions.tone,
              },
              {
                id: 'urgency',
                declaredType: 'score',
                instructions: inputs.questions.urgency.instructions,
                criteria: inputs.questions.urgency.criteria,
                rawQuestion: inputs.questions.urgency,
              },
            ],
          },
        ],
      },
      answers: [
        {
          kind: 'noul',
          id: 'billing',
          rawAnswer: outputs.answers.billing,
          probabilityTrue: 0.98,
        },
        {
          kind: 'choice',
          id: 'tone',
          rawAnswer: outputs.answers.tone,
          choice: 'urgent',
          confidence: 0.84,
          probabilities: [
            { label: 'calm', probability: 0.16 },
            { label: 'urgent', probability: 0.84 },
          ],
        },
        {
          kind: 'score',
          id: 'urgency',
          rawAnswer: outputs.answers.urgency,
          score: 1.7,
          confidence: 0.72,
          levels: [
            { score: 0, description: 'Can wait', probability: 0.05 },
            { score: 1, description: { label: 'This week' }, probability: 0.2 },
            { score: 2, description: ['Today'], probability: 0.75 },
          ],
          range: { min: 0, max: 2 },
        },
      ],
    });
  });

  test('preserves every input question in declaration order', () => {
    const inputs = {
      state: 'Support conversation',
      model: 'jev-latest',
      evaluation_context: { customerTier: 'enterprise' },
      questions: {
        answered: { type: 'noul', instructions: 'Was the issue resolved?' },
        unanswered: {
          type: 'choice',
          instructions: { prompt: 'What should happen next?' },
          criteria: ['reply', 'escalate'],
        },
        malformed: 'keep the original value',
      },
      extra_body: { temperature: 0.2 },
    };

    const result = translate({
      inputs,
      outputs: { answers: { answered: { type: 'noul', noul: 0.75 } } },
    });

    expect(result?.inputs.fields[1]).toEqual({
      kind: 'questions',
      fieldKey: 'questions',
      items: [
        {
          id: 'answered',
          declaredType: 'noul',
          instructions: 'Was the issue resolved?',
          rawQuestion: inputs.questions.answered,
        },
        {
          id: 'unanswered',
          declaredType: 'choice',
          instructions: inputs.questions.unanswered.instructions,
          criteria: inputs.questions.unanswered.criteria,
          rawQuestion: inputs.questions.unanswered,
        },
        {
          id: 'malformed',
          rawQuestion: inputs.questions.malformed,
        },
      ],
    });
    expect(result?.answers[0]).not.toHaveProperty('question');
  });

  test('falls back to the normal questions field for a malformed question map', () => {
    const inputs = {
      model: { unexpected: 'object' },
      questions: 'not-a-question-map',
    };

    const result = translate({
      inputs,
      outputs: { answers: { valid: { type: 'noul', noul: 0.75 } } },
    });

    expect(result?.inputs.fields).toEqual([
      { kind: 'field', fieldKey: 'state' },
      { kind: 'field', fieldKey: 'questions' },
    ]);
    expect(result?.answers).toHaveLength(1);
  });

  test('derives the score range from the union of serialized legend and probability keys', () => {
    const result = translate({
      inputs: { questions: {} },
      outputs: {
        answers: {
          rating: {
            type: 'score',
            score: 2.5,
            confidence: 1,
            legend: { '1': 'Low', '3': 'High' },
            probabilities: { '0': 0, '1': 0.25, '3': 0.75, '4': 0 },
          },
        },
      },
    });

    expect(result?.answers[0]).toMatchObject({
      kind: 'score',
      score: 2.5,
      range: { min: 0, max: 4 },
      levels: [
        { score: 0, probability: 0 },
        { score: 1, description: 'Low', probability: 0.25 },
        { score: 3, description: 'High', probability: 0.75 },
        { score: 4, probability: 0 },
      ],
    });
  });

  test('preserves malformed and future answers without dropping valid siblings', () => {
    const malformed = { type: 'choice', choice: 'yes', confidence: 2, probabilities: { yes: 1 } };
    const future = { type: 'ranking', order: ['a', 'b'] };
    const result = translate({
      inputs: {
        questions: {
          malformed: 'not-an-object',
          future: { type: 'ranking', instructions: 'Rank these' },
        },
      },
      outputs: {
        answers: {
          valid: { type: 'noul', noul: 0.4 },
          malformed,
          future,
          missingType: { value: true },
        },
      },
    });

    expect(result?.answers).toEqual([
      {
        kind: 'noul',
        id: 'valid',
        rawAnswer: { type: 'noul', noul: 0.4 },
        probabilityTrue: 0.4,
      },
      {
        kind: 'unknown',
        id: 'malformed',
        rawAnswer: malformed,
        declaredType: 'choice',
        reason: 'malformed',
      },
      {
        kind: 'unknown',
        id: 'future',
        rawAnswer: future,
        declaredType: 'ranking',
        reason: 'unsupported',
      },
      {
        kind: 'unknown',
        id: 'missingType',
        rawAnswer: { value: true },
        declaredType: undefined,
        reason: 'malformed',
      },
    ]);
  });

  test.each([
    { type: 'noul', noul: -0.1 },
    { type: 'noul', noul: Number.NaN },
    { type: 'choice', choice: 'a', confidence: 0.5, probabilities: { a: 1.01 } },
    { type: 'choice', choice: 'a', confidence: Number.POSITIVE_INFINITY, probabilities: { a: 1 } },
    {
      type: 'score',
      score: 3,
      confidence: 0.5,
      legend: { '0': 'Low', '2': 'High' },
      probabilities: { '0': 0, '2': 1 },
    },
    { type: 'score', score: 1, confidence: 0.5, legend: { low: 'Low' }, probabilities: { '1': 1 } },
    { type: 'score', score: 1, confidence: 0.5, legend: {}, probabilities: {} },
  ])('treats out-of-range or non-finite known answers as malformed: %p', (answer) => {
    const result = translate({ inputs: {}, outputs: { answers: { answer } } });
    expect(result?.answers[0]).toMatchObject({ kind: 'unknown', declaredType: answer.type, reason: 'malformed' });
  });

  test.each([
    { outputs: undefined },
    { outputs: null },
    { outputs: { result: 'custom response without answers' } },
    { outputs: { answers: [] } },
    { outputs: { answers: {} } },
  ])('returns null when outputs do not contain a standard non-empty answers map: %p', ({ outputs }) => {
    expect(translate({ inputs: {}, outputs })).toBeNull();
  });
});

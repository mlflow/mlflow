import { describe, expect, it } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { DecisionAnswers, DecisionInputs } from './DecisionView';
import { resolveOpenAIDecision } from './decision-utils/openai';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('OpenAI Decisions view', () => {
  const resolve = (inputs: unknown, outputs: unknown) =>
    resolveOpenAIDecision({ chatMessageFormat: 'openai_decisions', inputs, outputs });

  it('requires the Decisions marker and a valid array of answers', () => {
    const inputs = {
      input: 'Review the answer',
      questions: [{ type: 'predicate', name: 'correctness', instructions: 'Is it correct?' }],
    };
    const outputs = { answers: [{ type: 'predicate', name: 'correctness', probability: 0.8 }] };
    expect(resolveOpenAIDecision({ chatMessageFormat: 'openai', outputs })).toBeNull();
    expect(resolve(inputs, { answers: {} })).toBeNull();
    expect(resolve(inputs, { answers: [] })).toBeNull();
    expect(resolve({}, outputs)).toBeNull();
    expect(resolve(inputs, outputs)?.answers).toEqual([{ id: 'correctness', value: outputs.answers[0] }]);
  });

  it('uses the same question and answer cards for predicates, typed choices, scores, and refusals', async () => {
    const inputs = {
      input: 'Review this agent response',
      model: 'gpt-6-luna',
      questions: [
        { type: 'predicate', name: 'correctness', instructions: 'Is the answer correct?' },
        {
          type: 'choice',
          instructions: 'Which value was returned?',
          choices: [
            { value: true, description: 'Boolean' },
            { value: 'true', description: 'Text' },
          ],
        },
        {
          type: 'score',
          name: 'quality',
          instructions: 'Rate the quality',
          levels: [{ label: 'Low' }, { label: 'High' }],
        },
        { type: 'predicate', name: 'policy', instructions: 'Is this allowed?' },
      ],
    };
    const outputs = {
      answers: [
        { type: 'predicate', name: 'correctness', probability: 0.37 },
        {
          type: 'choice',
          name: null,
          choice: 'true',
          confidence: 0.8,
          probabilities: [
            { value: true, probability: 0.2 },
            { value: 'true', probability: 0.8 },
          ],
        },
        {
          type: 'score',
          name: 'quality',
          score: 0.9,
          confidence: 0.9,
          probabilities: [
            { value: 0, label: 'Low', probability: 0.1 },
            { value: 1, label: 'High', probability: 0.9 },
          ],
        },
        { type: 'refusal', name: 'policy' },
      ],
      usage: { input_tokens: 100 },
    };
    const decision = resolve(inputs, outputs);
    expect(decision).not.toBeNull();

    render(
      <>
        <DecisionInputs
          decision={decision!}
          fields={Object.entries(inputs).map(([key, value]) => ({ key, value: JSON.stringify(value) }))}
        />
        <DecisionAnswers decision={decision!} />
      </>,
      { wrapper: Wrapper },
    );

    expect(screen.getByText('Review this agent response')).toBeInTheDocument();
    expect(screen.queryByText('gpt-6-luna')).not.toBeInTheDocument();
    const questions = within(screen.getByTestId('decision-questions'));
    expect(questions.getAllByRole('listitem')).toHaveLength(4);
    expect(questions.getByText('correctness').closest('[role="listitem"]')).toHaveTextContent('Is the answer correct?');
    expect(questions.getByText('Question 2').closest('summary')).toHaveTextContent('Which value was returned?');
    await userEvent.click(questions.getByText('correctness').closest('summary') as HTMLElement);
    expect(
      within(questions.getByText('correctness').closest('details') as HTMLElement).getByText('Instructions'),
    ).toBeInTheDocument();
    await userEvent.click(questions.getByText('Question 2').closest('summary') as HTMLElement);
    expect(questions.getByText('Choices')).toBeInTheDocument();
    await userEvent.click(questions.getByText('quality').closest('summary') as HTMLElement);
    expect(questions.getByText('Levels')).toBeInTheDocument();

    const answers = within(screen.getByTestId('decision-answers'));
    expect(answers.getAllByRole('listitem')).toHaveLength(4);
    const summary = (id: string) => answers.getByText(id).closest('summary') as HTMLElement;
    expect(summary('correctness')).toHaveTextContent('Predicate');
    expect(summary('correctness')).toHaveTextContent('37%');
    expect(summary('correctness')).toHaveTextContent('probability of true');
    expect(summary('correctness')).not.toHaveTextContent('false');
    expect(summary('Question 2')).toHaveTextContent('"true"');
    expect(summary('Question 2')).toHaveTextContent('80% confidence');
    expect(summary('quality')).toHaveTextContent('Range 0–1');
    expect(summary('quality')).toHaveTextContent('90% confidence');
    expect(summary('policy')).toHaveTextContent('Refused');
    expect(answers.queryByText('input_tokens')).not.toBeInTheDocument();

    await userEvent.click(summary('correctness'));
    const predicateDetails = within(summary('correctness').closest('details') as HTMLElement);
    expect(predicateDetails.getByRole('progressbar', { name: 'Probability for true' })).toHaveAttribute(
      'aria-valuenow',
      '37',
    );
    expect(predicateDetails.getByRole('progressbar', { name: 'Probability for false' })).toHaveAttribute(
      'aria-valuenow',
      '63',
    );
    await userEvent.click(summary('Question 2'));
    const choiceDetails = within(summary('Question 2').closest('details') as HTMLElement);
    expect(choiceDetails.getByRole('progressbar', { name: 'Probability for "true"' })).toHaveAttribute(
      'aria-valuenow',
      '80',
    );
    await userEvent.click(summary('quality'));
    const scoreDetails = within(summary('quality').closest('details') as HTMLElement);
    expect(scoreDetails.getByRole('progressbar', { name: 'Probability for score 1' })).toHaveAttribute(
      'aria-valuenow',
      '90',
    );
    expect(answers.getByText('High')).toBeInTheDocument();
    await userEvent.click(summary('policy'));
    expect(
      within(summary('policy').closest('details') as HTMLElement).getByText(/"type": "refusal"/),
    ).toBeInTheDocument();
  });

  const resolveAnswer = (question: unknown, answer: unknown) =>
    resolve({ input: 'Review the answer', questions: [question] }, { answers: [answer] });

  it('falls back to raw fields when an OpenAI answer violates its shape or question type', () => {
    const predicate = { type: 'predicate', instructions: 'Is this correct?' };
    const choice = { type: 'choice', instructions: 'Select one', choices: [{ value: 'yes' }, { value: 'no' }] };
    const score = { type: 'score', instructions: 'Rate it', levels: [{ label: 'Low' }, { label: 'High' }] };

    expect(resolveAnswer(predicate, { type: 'predicate', probability: 1.2 })).toBeNull();
    expect(resolveAnswer(predicate, { type: 'future', probability: 0.5 })).toBeNull();
    expect(resolveAnswer(choice, { type: 'predicate', probability: 0.5 })).toBeNull();
    expect(
      resolveAnswer(choice, {
        type: 'choice',
        choice: 'yes',
        confidence: 0.8,
        probabilities: [{ value: 'no', probability: 1 }],
      }),
    ).toBeNull();
    expect(
      resolveAnswer(score, {
        type: 'score',
        score: 1,
        confidence: 0.8,
        probabilities: [{ value: 1, label: 'High', probability: -0.1 }],
      }),
    ).toBeNull();
    expect(resolveAnswer(predicate, null)).toBeNull();
    expect(
      resolveAnswer({ ...predicate, name: 'correctness' }, { type: 'predicate', name: 'other', probability: 0.5 }),
    ).toBeNull();
  });

  it('requires choice probabilities to match every typed option and sum to one', () => {
    const question = {
      type: 'choice',
      name: 'selection',
      instructions: 'Select one',
      choices: [{ value: true }, { value: 'true' }],
    };
    const answer = {
      type: 'choice',
      name: 'selection',
      choice: true,
      confidence: 0.6,
      probabilities: [
        { value: true, probability: 0.6 },
        { value: 'true', probability: 0.4 },
      ],
    };
    expect(resolveAnswer(question, answer)).not.toBeNull();
    expect(
      resolveAnswer(question, {
        ...answer,
        probabilities: [
          { value: true, probability: 0.6 },
          { value: 'true', probability: 0.395 },
        ],
      }),
    ).not.toBeNull();
    expect(
      resolveAnswer(question, {
        ...answer,
        probabilities: [
          { value: true, probability: 0.5 },
          { value: 'true', probability: 0.3 },
        ],
      }),
    ).toBeNull();
    expect(
      resolveAnswer(question, {
        ...answer,
        probabilities: [
          { value: true, probability: 0.6 },
          { value: 'other', probability: 0.4 },
        ],
      }),
    ).toBeNull();
    expect(
      resolveAnswer(
        { ...question, choices: [{ value: true }, { value: false }] },
        {
          ...answer,
          choice: 'true',
          probabilities: [
            { value: 'true', probability: 0.6 },
            { value: 'false', probability: 0.4 },
          ],
        },
      ),
    ).toBeNull();
    expect(resolveAnswer(question, { ...answer, probabilities: [{ value: true, probability: 1 }] })).toBeNull();
    expect(resolveAnswer(question, { ...answer, choice: 'other' })).toBeNull();
    expect(
      resolveAnswer(
        { ...question, choices: [{ value: true }, { value: true }] },
        { type: 'refusal', name: 'selection' },
      ),
    ).toBeNull();
  });

  it('requires score probabilities to match level indices and labels and sum to one', () => {
    const question = {
      type: 'score',
      name: 'quality',
      instructions: 'Rate quality',
      levels: [{ label: 'Low' }, { label: 'High' }],
    };
    const answer = {
      type: 'score',
      name: 'quality',
      score: 0.7,
      confidence: 0.7,
      probabilities: [
        { value: 0, label: 'Low', probability: 0.3 },
        { value: 1, label: 'High', probability: 0.7 },
      ],
    };
    expect(resolveAnswer(question, answer)).not.toBeNull();
    expect(
      resolveAnswer(question, {
        ...answer,
        probabilities: [
          { value: 0, label: 'Low', probability: 0.3 },
          { value: 1, label: 'High', probability: 0.5 },
        ],
      }),
    ).toBeNull();
    expect(
      resolveAnswer(question, {
        ...answer,
        probabilities: [
          { value: 0, label: 'Low', probability: 0.3 },
          { value: 2, label: 'High', probability: 0.7 },
        ],
      }),
    ).toBeNull();
    expect(
      resolveAnswer(question, {
        ...answer,
        probabilities: [
          { value: 0, label: 'Low', probability: 0.3 },
          { value: 1, label: 'Medium', probability: 0.7 },
        ],
      }),
    ).toBeNull();
    expect(
      resolveAnswer(question, { ...answer, probabilities: [{ value: 0, label: 'Low', probability: 1 }] }),
    ).toBeNull();
    expect(resolveAnswer(question, { ...answer, score: 1.7 })).toBeNull();
  });
});

import { describe, expect, it } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { resolveTypeSafeDecision, TypeSafeDecisionAnswers, TypeSafeDecisionInputs } from './TypeSafeDecisionView';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const resolve = (inputs: unknown, outputs: unknown) =>
  resolveTypeSafeDecision({ chatMessageFormat: 'typesafe', inputs, outputs });

describe('TypeSafe decision view', () => {
  it('uses the exact span marker and falls back for custom responses without answers', () => {
    const outputs = { answers: { safety: { type: 'noul', noul: 0.9 } } };
    expect(resolveTypeSafeDecision({ chatMessageFormat: 'TypeSafe', outputs })).toBeNull();
    expect(resolveTypeSafeDecision({ chatMessageFormat: 'openai', outputs })).toBeNull();
    expect(resolve({}, { result: 'custom response' })).toBeNull();
    expect(resolve({}, { answers: {} })).toBeNull();
    expect(resolve({}, { answers: [] })).toBeNull();
    expect(resolve({}, outputs)?.answers).toHaveLength(1);
  });

  it('shows only state and questions in Pretty inputs', async () => {
    const inputs = {
      state: 'I was charged twice',
      questions: {
        billing: { type: 'noul', instructions: 'Is this about billing?', criteria: { true: 'A charge is mentioned' } },
        followup: { type: 'choice', instructions: 'What should happen next?' },
      },
      model: 'jev-latest',
      temperature: 0,
    };
    const decision = resolve(inputs, { answers: { billing: { type: 'noul', noul: 0.98 } } });
    expect(decision).not.toBeNull();

    render(
      <TypeSafeDecisionInputs
        decision={decision!}
        fields={Object.entries(inputs).map(([key, value]) => ({ key, value: JSON.stringify(value) }))}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByText('I was charged twice')).toBeInTheDocument();
    expect(screen.getByText('questions')).toBeInTheDocument();
    expect(screen.queryByText('jev-latest')).not.toBeInTheDocument();
    expect(screen.queryByText('temperature')).not.toBeInTheDocument();
    const questions = within(screen.getByTestId('decision-questions'));
    expect(questions.getAllByRole('listitem')).toHaveLength(2);
    expect(questions.getByText('billing').closest('summary')).toHaveTextContent('Is this about billing?');
    await userEvent.click(questions.getByText('billing').closest('summary') as HTMLElement);
    expect(questions.getByText('Criteria')).toBeInTheDocument();
    expect(questions.getAllByText('Is this about billing?')).toHaveLength(1);
    expect(questions.getAllByText('What should happen next?')).toHaveLength(1);
  });

  it('shows Noul, choice, and score cards with accessible probability bars and inline score labels', async () => {
    const decision = resolve(
      {},
      {
        answers: {
          billing: { type: 'noul', noul: 0.98 },
          tone: { type: 'choice', choice: 'urgent', confidence: 0.84, probabilities: { calm: 0.16, urgent: 0.84 } },
          urgency: {
            type: 'score',
            score: 1.7,
            confidence: 0.72,
            legend: { 0: 'Can wait', 1: { label: 'This week' }, 2: 'Today' },
            probabilities: { 0: 0.05, 1: 0.2, 2: 0.75 },
          },
        },
        model: 'not shown in Pretty',
      },
    );
    expect(decision?.answers).toHaveLength(3);

    render(<TypeSafeDecisionAnswers decision={decision!} />, { wrapper: Wrapper });
    const answers = within(screen.getByTestId('decision-answers'));
    const summary = (id: string) => answers.getByText(id).closest('summary') as HTMLElement;
    expect(answers.getAllByRole('listitem')).toHaveLength(3);
    expect(summary('billing')).toHaveTextContent('true');
    expect(summary('billing')).toHaveTextContent('98% probability');
    expect(summary('tone')).toHaveTextContent('urgent');
    expect(summary('tone')).toHaveTextContent('84% confidence');
    expect(summary('urgency')).toHaveTextContent('1.7');
    expect(summary('urgency')).toHaveTextContent('Range 0–2');
    expect(summary('urgency')).toHaveTextContent('72% confidence');
    expect(answers.queryByText('not shown in Pretty')).not.toBeInTheDocument();

    await userEvent.click(summary('billing'));
    expect(summary('billing').closest('details')).toHaveAttribute('open');
    expect(answers.getByText('False')).toBeInTheDocument();
    expect(answers.getByText('2%')).toBeInTheDocument();
    expect(answers.getByRole('progressbar', { name: 'Probability for true' })).toHaveAttribute('aria-valuenow', '98');
    await userEvent.click(summary('tone'));
    expect(answers.getByText('calm')).toBeInTheDocument();
    expect(answers.getByText('16%')).toBeInTheDocument();
    expect(answers.getByRole('progressbar', { name: 'Probability for calm' })).toHaveAttribute('aria-valuenow', '16');
    expect(within(summary('tone').closest('details') as HTMLElement).getByText('Confidence')).toBeInTheDocument();
    await userEvent.click(summary('urgency'));
    expect(answers.getByText('2')).toBeInTheDocument();
    expect(answers.getByText('75%')).toBeInTheDocument();
    expect(answers.getByRole('progressbar', { name: 'Probability for score 2' })).toHaveAttribute(
      'aria-valuenow',
      '75',
    );
    expect(answers.getByText('Today')).toBeInTheDocument();
    expect(answers.getByText(/"label": "This week"/)).toBeInTheDocument();
    expect(answers.queryByText('Legend')).not.toBeInTheDocument();
  });

  it('shows malformed answers raw when score legend and probability keys differ', async () => {
    const decision = resolve(
      {},
      {
        answers: {
          brokenChoice: { type: 'choice', choice: 'yes', confidence: 0.7, probabilities: { yes: 'unknown' } },
          brokenScore: {
            type: 'score',
            score: 1,
            confidence: 0.8,
            legend: { 0: 'Low', 1: 'High' },
            probabilities: { 0: 0.1, 2: 0.9 },
          },
          future: { type: 'ranking', order: ['a', 'b'] },
        },
      },
    );
    expect(decision?.answers).toHaveLength(3);

    render(<TypeSafeDecisionAnswers decision={decision!} />, { wrapper: Wrapper });
    const answers = within(screen.getByTestId('decision-answers'));
    const summary = (id: string) => answers.getByText(id).closest('summary') as HTMLElement;
    expect(summary('brokenChoice')).toHaveTextContent('Raw answer');
    expect(summary('future')).toHaveTextContent('Raw answer');
    expect(summary('brokenScore')).toHaveTextContent('Raw answer');
    await userEvent.click(summary('brokenScore'));
    const brokenScoreDetails = within(summary('brokenScore').closest('details') as HTMLElement);
    expect(brokenScoreDetails.getByText(/"type": "score"/)).toBeInTheDocument();
    expect(brokenScoreDetails.getByText(/"legend"/)).toBeInTheDocument();
    expect(brokenScoreDetails.getByText(/"probabilities"/)).toBeInTheDocument();
    expect(answers.queryByRole('progressbar')).not.toBeInTheDocument();
    await userEvent.click(summary('brokenChoice'));
    expect(
      within(summary('brokenChoice').closest('details') as HTMLElement).getByText(/"type": "choice"/),
    ).toBeInTheDocument();
  });
});

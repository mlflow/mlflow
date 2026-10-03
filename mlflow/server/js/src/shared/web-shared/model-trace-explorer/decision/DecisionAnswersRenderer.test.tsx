import { describe, expect, it } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { DecisionAnswersRenderer } from './DecisionAnswersRenderer';
import type { DecisionAnswerViewModel } from './Decision.types';

const answers: DecisionAnswerViewModel[] = [
  {
    kind: 'choice',
    id: 'category',
    rawAnswer: {},
    choice: 'billing',
    confidence: 0.61,
    probabilities: [
      { label: 'billing', probability: 0.74 },
      { label: 'other', probability: 0.26 },
    ],
  },
  {
    kind: 'score',
    id: 'urgency',
    rawAnswer: {},
    score: 1.74,
    confidence: 0.61,
    levels: [
      { score: 0, description: 'routine', probability: 0 },
      { score: 1, description: { label: 'soon' }, probability: 0.26 },
      { score: 2, description: 'urgent', probability: 0.74 },
    ],
    range: { min: 0, max: 2 },
  },
  {
    kind: 'noul',
    id: 'duplicate_charge',
    rawAnswer: {},
    probabilityTrue: 0.99,
  },
  {
    kind: 'unknown',
    id: 'future_answer',
    rawAnswer: { type: 'future', value: 42 },
    declaredType: 'future',
    reason: 'unsupported',
  },
];

const renderRenderer = () =>
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">
        <DecisionAnswersRenderer answers={answers} />
      </IntlProvider>
    </DesignSystemProvider>,
  );

describe('DecisionAnswersRenderer', () => {
  it('renders compact summaries for every supported answer type', () => {
    renderRenderer();

    expect(screen.getByText('billing')).toBeInTheDocument();
    expect(screen.getByText('1.74')).toBeInTheDocument();
    expect(screen.getByText('Range 0–2')).toBeInTheDocument();
    expect(screen.getByText('true')).toBeInTheDocument();
    expect(screen.getByText('99% probability')).toBeInTheDocument();
    expect(screen.getByText('Unsupported answer')).toBeInTheDocument();
    expect(screen.queryByText('Probability distribution')).not.toBeInTheDocument();
  });

  it('expands answer details accessibly and gives Noul the same value and metric hierarchy as Choice', async () => {
    renderRenderer();

    const choiceToggle = screen.getByRole('button', { name: /^category\b/ });
    expect(choiceToggle).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(choiceToggle);

    expect(choiceToggle).toHaveAttribute('aria-expanded', 'true');
    expect(choiceToggle).toHaveAttribute('aria-controls');
    expect(screen.getByRole('region', { name: /^category\b/ })).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Probability for billing' })).toHaveAttribute('aria-valuenow', '74');
    expect(
      within(choiceToggle.closest('[role="listitem"]') as HTMLElement).getByText('Confidence'),
    ).toBeInTheDocument();

    const scoreToggle = screen.getByRole('button', { name: /^urgency\b/ });
    await userEvent.click(scoreToggle);
    expect(screen.getByText(/"label": "soon"/)).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Probability for score 2' })).toHaveAttribute('aria-valuenow', '74');

    const noulToggle = screen.getByRole('button', { name: /^duplicate_charge\b/ });
    await userEvent.click(noulToggle);
    const noulRow = noulToggle.closest('[role="listitem"]') as HTMLElement;
    expect(within(noulRow).getByRole('progressbar', { name: 'Probability for true' })).toHaveAttribute(
      'aria-valuenow',
      '99',
    );
    expect(within(noulToggle).getByText('true')).toBeInTheDocument();
    expect(within(noulToggle).getByText('99% probability')).toBeInTheDocument();
    expect(within(noulToggle).queryByText(/confidence/i)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /^future_answer\b/ }));
    expect(screen.getByText(/"type": "future"/)).toBeInTheDocument();
  });
});

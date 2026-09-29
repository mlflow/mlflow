import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { DecisionQuestionsRenderer } from './DecisionQuestionsRenderer';
import type { DecisionQuestionViewModel } from './Decision.types';

const questions: DecisionQuestionViewModel[] = [
  {
    id: 'category',
    declaredType: 'choice',
    instructions: 'Classify this request',
    rawQuestion: {},
  },
  {
    id: 'urgency',
    declaredType: 'score',
    instructions: { task: 'Rate the urgency', scale: [0, 1, 2] },
    criteria: { focus: 'customer impact' },
    rawQuestion: {},
  },
  {
    id: 'future_question',
    rawQuestion: null,
  },
];

const renderRenderer = (value: DecisionQuestionViewModel[] = questions) =>
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">
        <DecisionQuestionsRenderer questions={value} />
      </IntlProvider>
    </DesignSystemProvider>,
  );

describe('DecisionQuestionsRenderer', () => {
  it('renders ordered question summaries without adding a synthetic section title', () => {
    renderRenderer();

    const questionIds = screen
      .getAllByText(/^(category|urgency|future_question)$/)
      .map((element) => element.textContent);
    expect(questionIds).toEqual(['category', 'urgency', 'future_question']);
    expect(screen.getByText('choice')).toBeInTheDocument();
    expect(screen.getByText('score')).toBeInTheDocument();
    expect(screen.getByText('Unknown type')).toBeInTheDocument();
    expect(screen.getByText('Classify this request')).toBeInTheDocument();
    expect(screen.queryByText('Questions')).not.toBeInTheDocument();
  });

  it('uses a native disclosure for structured instructions and criteria', async () => {
    const { container } = renderRenderer();
    const urgency = screen.getByText('urgency');
    const disclosure = urgency.closest('details');

    expect(disclosure).not.toHaveAttribute('open');
    expect(screen.getByText('Criteria')).not.toBeVisible();

    await userEvent.click(urgency.closest('summary') as HTMLElement);

    expect(disclosure).toHaveAttribute('open');
    expect(screen.getByText('Instructions')).toBeVisible();
    expect(screen.getByText(/"task": "Rate the urgency"/)).toBeVisible();
    expect(screen.getByText(/"focus": "customer impact"/)).toBeVisible();
    expect(container.querySelectorAll('details')).toHaveLength(1);
  });

  it('returns no markup when there are no questions', () => {
    const { container } = renderRenderer([]);

    expect(container).toBeEmptyDOMElement();
  });
});

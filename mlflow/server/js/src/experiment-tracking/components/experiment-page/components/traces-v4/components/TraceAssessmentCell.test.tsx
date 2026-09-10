import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { TraceAssessmentCell, TraceAssessmentHoverContent } from './TraceAssessmentCell';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceAssessmentCell', () => {
  test('renders the most recent assessment value as a tag', () => {
    const trace = makeTrace('tr-0', { assessments: [makeFeedbackAssessment('relevance', 'yes')] });
    renderWithProviders(<TraceAssessmentCell trace={trace} assessmentName="relevance" columnType="categorical" />);
    // 'yes' feedback renders as a "Yes" tag (AssessmentDisplayValue).
    expect(screen.getByText('Yes')).toBeInTheDocument();
  });

  test('renders nothing when the trace has no assessment of that name', () => {
    const trace = makeTrace('tr-0', { assessments: [] });
    const { container } = renderWithProviders(
      <TraceAssessmentCell trace={trace} assessmentName="relevance" columnType="categorical" />,
    );
    // eslint-disable-next-line testing-library/no-node-access -- asserting the cell renders empty
    expect(container.firstChild).toBeNull();
  });

  test('the trigger tag shows no redundant value tooltip on hover (the hover card covers the value)', async () => {
    const trace = makeTrace('tr-0', {
      assessments: [makeFeedbackAssessment('relevance', 'yes', { rationale: 'On topic.' })],
    });
    renderWithProviders(<TraceAssessmentCell trace={trace} assessmentName="relevance" columnType="categorical" />);

    // Hovering the trigger opens the detailed hover card (its Rationale section appears)…
    await userEvent.hover(screen.getByText('Yes'));
    expect(await screen.findByText('Rationale')).toBeInTheDocument();
    // …and no separate value tooltip renders on top of it (the redundant tooltip was removed).
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  test('renders numeric score bar for numeric column with number value', () => {
    const trace = makeTrace('tr-0', { assessments: [makeFeedbackAssessment('score', 0.75)] });
    renderWithProviders(<TraceAssessmentCell trace={trace} assessmentName="score" columnType="numeric" />);
    // Score bar displays the value formatted to 2 decimals
    expect(screen.getByText('0.75')).toBeInTheDocument();
  });

  test('renders score bar with 0 value (not a tag)', () => {
    const trace = makeTrace('tr-0', { assessments: [makeFeedbackAssessment('score', 0)] });
    renderWithProviders(<TraceAssessmentCell trace={trace} assessmentName="score" columnType="numeric" />);
    // Score bar displays 0.00
    expect(screen.getByText('0.00')).toBeInTheDocument();
  });

  test('error tag in numeric column (error takes precedence)', () => {
    const trace = makeTrace('tr-0', {
      assessments: [
        makeFeedbackAssessment('score', 0.75, {
          feedback: { value: null, error: { error_code: 'JUDGE_ERROR', error_message: 'Model failed' } },
        }),
      ],
    });
    renderWithProviders(<TraceAssessmentCell trace={trace} assessmentName="score" columnType="numeric" />);
    // Error tag renders instead of the score bar
    expect(screen.queryByText('0.75')).not.toBeInTheDocument();
  });
});

describe('TraceAssessmentHoverContent', () => {
  test('renders rationale, source id, and a relative last-updated time', () => {
    const assessment = makeFeedbackAssessment('relevance', 'yes', {
      rationale: 'The response is on topic.',
      source: { source_type: 'HUMAN', source_id: 'alice' },
    });
    renderWithProviders(<TraceAssessmentHoverContent assessment={assessment} />);

    expect(screen.getByText('Rationale')).toBeInTheDocument();
    expect(screen.getByText('The response is on topic.')).toBeInTheDocument();
    expect(screen.getByText('Source')).toBeInTheDocument();
    // AssessmentSourceName shows the source id.
    expect(screen.getByText('alice')).toBeInTheDocument();
    // The last-updated section renders (FormattedRelativeTime output is locale/time dependent).
    expect(screen.getByText(/Updated/)).toBeInTheDocument();
  });

  test('omits the Rationale section when the assessment has none', () => {
    const assessment = makeFeedbackAssessment('relevance', 'yes', { rationale: undefined });
    renderWithProviders(<TraceAssessmentHoverContent assessment={assessment} />);
    expect(screen.queryByText('Rationale')).not.toBeInTheDocument();
    // The source section still renders.
    expect(screen.getByText('Source')).toBeInTheDocument();
  });
});

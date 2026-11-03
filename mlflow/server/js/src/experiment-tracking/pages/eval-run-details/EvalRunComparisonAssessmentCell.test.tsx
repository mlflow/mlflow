import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { createIntl } from 'react-intl';
import { EvalRunComparisonTestWrapper } from './EvalRunComparisonTestProviders';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import {
  EvalRunComparisonAssessmentCell,
  formatEvalRunComparisonAssessmentDelta,
  type EvalRunComparisonAssessmentCellProps,
} from './EvalRunComparisonAssessmentCell';

const INCREASE = { value: 0.7, baselineValue: 0.65, difference: 0.7 - 0.65 };
const DECREASE = { value: 0.6, baselineValue: 0.8, difference: 0.6 - 0.8 };

const renderCell = (props: Partial<EvalRunComparisonAssessmentCellProps> = {}) =>
  render(
    <EvalRunComparisonAssessmentCell
      trace={makeTrace('candidate', {
        assessments: [makeFeedbackAssessment('quality', props.delta?.value ?? INCREASE.value)],
      })}
      assessmentName="quality"
      {...props}
    />,
    { wrapper: EvalRunComparisonTestWrapper },
  );

describe('formatEvalRunComparisonAssessmentDelta', () => {
  const intl = createIntl({ locale: 'en', messages: {} });

  test.each([
    { difference: 0.05, expected: '+0.05' },
    { difference: -0.2, expected: '-0.20' },
    { difference: 7.1, expected: '+7.10' },
    { difference: 0.123, expected: '+0.12' },
    { difference: 1, expected: '+1.00' },
  ])('formats $difference as $expected', ({ difference, expected }) => {
    expect(formatEvalRunComparisonAssessmentDelta(intl, difference)).toBe(expected);
  });
});

describe('EvalRunComparisonAssessmentCell', () => {
  test('shows the score followed by its difference from the baseline', () => {
    renderCell({ delta: INCREASE });

    expect(screen.getByText('0.70')).toBeVisible();
    expect(screen.getByText('+0.05')).toBeVisible();
  });

  test('shows a decrease as a negative difference', () => {
    renderCell({ delta: DECREASE });

    expect(screen.getByText('-0.20')).toBeVisible();
  });

  test('shows only the score when there is no delta', () => {
    renderCell();

    expect(screen.getByText('0.70')).toBeVisible();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  test('shows the baseline and current scores with the difference when hovering the pill', async () => {
    renderCell({ delta: INCREASE });

    await userEvent.hover(screen.getByText('+0.05'));

    expect(await screen.findByText('0.65 → 0.70')).toBeVisible();
    expect(screen.getAllByText('+0.05')).toHaveLength(2);
  });
});

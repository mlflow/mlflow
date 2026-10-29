import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { TraceAssessmentCell } from './TraceAssessmentCell';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import {
  PAIRWISE_PREFERENCE_ASSESSMENT_NAME,
  PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY,
  PAIRWISE_SCORE_WIN_VALUE,
} from '../utils/pairwiseReview';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const trace = makeTrace('tr-0', {
  assessments: [
    makeFeedbackAssessment(PAIRWISE_PREFERENCE_ASSESSMENT_NAME, PAIRWISE_SCORE_WIN_VALUE, {
      source: { source_type: 'HUMAN', source_id: 'User' },
      metadata: { [PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY]: 'tr-1' },
    }),
  ],
});

const renderCell = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TraceAssessmentCell trace={trace} assessmentName={PAIRWISE_PREFERENCE_ASSESSMENT_NAME} columnType="numeric" />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceAssessmentCell with pairwise preferences', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test('shows the win count with the new eval runs UI', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderCell();
    expect(screen.getByText('Win')).toBeInTheDocument();
  });

  test('shows the raw value without the new eval runs UI', () => {
    renderCell();
    expect(screen.queryByText('Win')).not.toBeInTheDocument();
    expect(screen.getByText('1.00')).toBeInTheDocument();
  });
});

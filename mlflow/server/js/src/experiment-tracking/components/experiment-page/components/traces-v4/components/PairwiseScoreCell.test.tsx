import { describe, expect, test } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';

import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import {
  PAIRWISE_PREFERENCE_ASSESSMENT_NAME,
  PAIRWISE_SCORE_LOSE_VALUE,
  PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY,
  PAIRWISE_SCORE_WIN_VALUE,
} from '../utils/pairwiseReview';
import { PairwiseScoreCell } from './PairwiseScoreCell';

describe('PairwiseScoreCell', () => {
  test('shows win and loss counts across active pairwise assessments', () => {
    const pairwiseAssessment = (id: string, value: number, opponentTraceId: string) =>
      makeFeedbackAssessment(PAIRWISE_PREFERENCE_ASSESSMENT_NAME, value, {
        assessment_id: id,
        source: { source_type: 'HUMAN', source_id: 'reviewer@example.com' },
        metadata: { [PAIRWISE_SCORE_OPPONENT_TRACE_ID_METADATA_KEY]: opponentTraceId },
      });
    const trace = makeTrace('tr-0', {
      assessments: [
        pairwiseAssessment('win-1', PAIRWISE_SCORE_WIN_VALUE, 'tr-1'),
        pairwiseAssessment('win-2', PAIRWISE_SCORE_WIN_VALUE, 'tr-2'),
        pairwiseAssessment('lose-1', PAIRWISE_SCORE_LOSE_VALUE, 'tr-3'),
        { ...pairwiseAssessment('invalid', PAIRWISE_SCORE_WIN_VALUE, 'tr-4'), valid: false },
      ],
    });

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <PairwiseScoreCell trace={trace} />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(screen.getByText('Win (2)')).toBeInTheDocument();
    expect(screen.getByText('Lose')).toBeInTheDocument();
  });
});

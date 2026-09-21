import { describe, expect, it } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import type { FeedbackAssessment } from '@databricks/web-shared/model-trace-explorer';
import JevAssessmentDetails from './JevAssessmentDetails';

const assessment: FeedbackAssessment = {
  assessment_id: 'a',
  assessment_name: 'relevance',
  trace_id: 'tr-1',
  source: { source_type: 'LLM_JUDGE', source_id: 'typesafe:/jev-latest' },
  create_time: '2026-01-01',
  last_update_time: '2026-01-01',
  feedback: { value: false },
};

describe('JevAssessmentDetails', () => {
  it('shows the original probability alongside a thresholded false decision', () => {
    renderWithDesignSystem(
      <JevAssessmentDetails
        assessment={{
          ...assessment,
          metadata: {
            'jev.model': 'jev-latest',
            'jev.probability': '0.65',
          },
        }}
      />,
    );
    expect(screen.getByText('Jev result: false')).toBeInTheDocument();
    expect(screen.getByText('Probability of true')).toBeInTheDocument();
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.queryByText('Confidence')).not.toBeInTheDocument();
  });

  it('shows ordered score probabilities with the legend and confidence', () => {
    renderWithDesignSystem(
      <JevAssessmentDetails
        assessment={{
          ...assessment,
          feedback: { value: 0.8 },
          metadata: {
            'jev.model': 'jev-latest',
            'jev.probabilities': '{"0":0.2,"1":0.8}',
            'jev.legend': '{"0":"Incorrect","1":"Correct"}',
            'jev.confidence': '0.6',
          },
        }}
      />,
    );
    expect(screen.getByText('0: Incorrect')).toBeInTheDocument();
    expect(screen.getByText('1: Correct')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();
    expect(screen.getByText('60%')).toBeInTheDocument();
  });

  it('ignores malformed probability metadata without losing the decision', () => {
    renderWithDesignSystem(
      <JevAssessmentDetails
        assessment={{
          ...assessment,
          metadata: {
            'jev.model': 'jev-latest',
            'jev.probabilities': 'invalid',
            'jev.probability': 'NaN',
          },
        }}
      />,
    );
    expect(screen.getByText('Jev result: false')).toBeInTheDocument();
    expect(screen.queryByText('Probability of true')).not.toBeInTheDocument();
  });
});

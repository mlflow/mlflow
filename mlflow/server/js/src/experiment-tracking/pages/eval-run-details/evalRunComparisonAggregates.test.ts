import { describe, expect, jest, test } from '@jest/globals';
import { createIntl } from 'react-intl';
import { convertTraceInfoV3ToRunEvalEntry, getAssessmentInfos } from '@databricks/web-shared/genai-traces-table';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { getEvalRunComparisonAssessmentAggregates } from './evalRunComparisonAggregates';
import { createComparisonTrace } from './evalRunComparisonTestUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

const intl = createIntl({ locale: 'en' });

const relevanceAverage = () => {
  const failed = createComparisonTrace('run', { relevance: 1 }, 1);
  const [assessment] = failed.assessments ?? [];
  if (!assessment) throw new Error('Missing fixture assessment');
  assessment.error = { error_code: 'FAILED', error_message: 'Scorer failed' };
  const entries = [createComparisonTrace('run', { relevance: 5 }, 0), failed].map((trace) =>
    convertTraceInfoV3ToRunEvalEntry(trace),
  );
  const [info] = getAssessmentInfos(intl, entries, []);
  return getEvalRunComparisonAssessmentAggregates(
    info,
    entries.map((currentRunValue) => ({ currentRunValue })),
  ).currentNumericAverage;
};

describe('getEvalRunComparisonAssessmentAggregates', () => {
  test('leaves an errored score out of the comparison average', () => {
    expect(relevanceAverage()).toBe(5);
  });

  test('keeps the shared aggregation unchanged with the new eval runs UI off', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValueOnce(false);
    expect(relevanceAverage()).toBe(3);
  });
});

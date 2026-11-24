import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';
import type { Assessment } from './ModelTrace.types';
import { getAssessmentMap } from './ModelTraceExplorer.utils';
import { MOCK_ASSESSMENT, MOCK_TRACE_INFO_V3 } from '../ModelTraceExplorer.test-utils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const PAIRWISE_ASSESSMENT: Assessment = {
  ...MOCK_ASSESSMENT,
  assessment_id: 'a-pairwise',
  assessment_name: 'Pairwise preference',
  source: { source_type: 'HUMAN', source_id: 'User' },
  feedback: { value: 1 },
  metadata: { 'mlflow.assessment.pairwiseOpponentTraceId': 'tr-other' },
};

const assessmentNames = (traceAssessments: Assessment[]) =>
  Object.values(getAssessmentMap({ ...MOCK_TRACE_INFO_V3, assessments: traceAssessments }))
    .flat()
    .map((assessment) => assessment.assessment_name);

describe('getAssessmentMap with pairwise preferences', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  it('leaves pairwise preferences out of the regular assessments with the new eval runs UI', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    expect(assessmentNames([MOCK_ASSESSMENT, PAIRWISE_ASSESSMENT])).toEqual(['Relevance']);
  });

  it('keeps every assessment without the new eval runs UI', () => {
    expect(assessmentNames([MOCK_ASSESSMENT, PAIRWISE_ASSESSMENT])).toEqual(['Relevance', 'Pairwise preference']);
  });
});

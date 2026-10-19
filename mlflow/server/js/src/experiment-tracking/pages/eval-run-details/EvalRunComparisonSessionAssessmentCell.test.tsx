import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { EvalRunComparisonTestWrapper as ProvidersWrapper } from './EvalRunComparisonTestProviders';
import { ASSESSMENT_SESSION_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { KnownEvaluationResultAssessmentStringValue } from '@databricks/web-shared/genai-traces-table';
import {
  makeFeedbackAssessment,
  makeTrace,
} from '../../components/experiment-page/components/traces-v4/test-utils/mockTraces';
import { EvalRunComparisonSessionAssessmentCell } from './EvalRunComparisonSessionAssessmentCell';
import type { SessionComparisonAssessmentInfo } from './evalRunComparisonSessionSummaries';

const renderSummary = (traces: ModelTraceInfoV3[], info: SessionComparisonAssessmentInfo) =>
  render(<EvalRunComparisonSessionAssessmentCell traces={traces} info={info} />, { wrapper: ProvidersWrapper });

// @ui-test-skill-generated
describe('EvalRunComparisonSessionAssessmentCell', () => {
  it.each([
    { dtype: 'numeric', values: [0.2, 0.6], text: '0.4 (AVG)' },
    { dtype: 'boolean', values: [false, false], text: '0% (TRUE)' },
    {
      dtype: 'pass-fail',
      values: [KnownEvaluationResultAssessmentStringValue.YES, KnownEvaluationResultAssessmentStringValue.NO],
      text: '50% (PASS)',
    },
  ] satisfies {
    dtype: SessionComparisonAssessmentInfo['dtype'];
    values: (number | boolean | string)[];
    text: string;
  }[])('shows the $dtype trace summary', ({ dtype, values, text }) => {
    renderSummary(
      values.map((value, index) =>
        makeTrace(`trace-${index}`, { assessments: [makeFeedbackAssessment('quality', value)] }),
      ),
      { name: 'quality', scope: 'trace', dtype },
    );
    expect(screen.getByText(text)).toBeVisible();
  });

  it('shows categorical trace value counts', () => {
    renderSummary(
      ['good', 'bad', 'good'].map((value, index) =>
        makeTrace(`trace-${index}`, { assessments: [makeFeedbackAssessment('quality', value)] }),
      ),
      { name: 'quality', scope: 'trace', dtype: 'string' },
    );
    expect(screen.getByText('good: 2')).toBeVisible();
    expect(screen.getByText('bad: 1')).toBeVisible();
  });

  it('shows an error when no trace score can contribute a value', () => {
    renderSummary(
      [
        makeTrace('failed', {
          assessments: [makeFeedbackAssessment('quality', 100, { error: { error_code: 'FAILED' } })],
        }),
      ],
      { name: 'quality', scope: 'trace', dtype: 'numeric' },
    );
    expect(screen.getByText('Error')).toBeVisible();
  });

  it('shows the recorded session score instead of a trace average', () => {
    renderSummary(
      [
        makeTrace('trace', {
          assessments: [
            makeFeedbackAssessment('quality', 0.2),
            makeFeedbackAssessment('quality', 0.9, { metadata: { [ASSESSMENT_SESSION_METADATA_KEY]: 'session' } }),
          ],
        }),
      ],
      { name: 'quality', scope: 'session', dtype: 'numeric' },
    );
    expect(screen.getByText('0.90')).toBeVisible();
    expect(screen.queryByText('0.2 (AVG)')).not.toBeInTheDocument();
  });

  it.each(['trace', 'session'] satisfies SessionComparisonAssessmentInfo['scope'][])(
    'shows a placeholder for absent %s feedback',
    (scope) => {
      renderSummary([makeTrace('trace')], { name: 'quality', scope, dtype: 'numeric' });
      expect(screen.getByText('—')).toBeVisible();
    },
  );
});

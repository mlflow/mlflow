import { filterEvaluationResults } from './EvaluationsFilterUtils';
import type { AssessmentFilter, EvalTraceComparisonEntry } from '../types';

describe('filterEvaluationResults', () => {
  const evals: EvalTraceComparisonEntry[] = [
    {
      currentRunValue: {
        evaluationId: `eval-00`,
        requestId: `req-00`,
        inputs: {
          messages: [
            { role: 'user', content: 'Hello one' },
            { role: 'assistant', content: 'Hello Response' },
          ],
        },
        inputsId: `inputs-00`,
        outputs: {},
        targets: {},
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no', rationale: 'Fails criteria' }],
        responseAssessmentsByName: {
          safety: [{ name: 'safety', stringValue: 'yes', rationale: 'Fails criteria' }],
          customAssessment: [{ name: 'customAssessment', stringValue: 'maybe', rationale: 'Unclear' }],
        },
        metrics: {},
      },
      otherRunValue: {
        evaluationId: `eval-01`,
        requestId: `req-01`,
        inputs: {
          messages: [
            { role: 'user', content: 'Hello one' },
            { role: 'assistant', content: 'Hello Response' },
          ],
        },
        inputsId: `inputs-01`,
        outputs: {},
        targets: {},
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes', rationale: 'Passes criteria' }],
        responseAssessmentsByName: {
          safety: [{ name: 'safety', stringValue: 'no', rationale: 'Passes criteria' }],
          customAssessment: [{ name: 'customAssessment', stringValue: 'yes', rationale: 'Clear' }],
        },
        metrics: {},
      },
    },
    {
      currentRunValue: {
        evaluationId: `eval-02`,
        requestId: `req-02`,
        inputs: {
          messages: [
            { role: 'user', content: 'Hello again' },
            { role: 'assistant', content: 'Hello again Response' },
          ],
        },
        inputsId: `inputs-02`,
        outputs: {},
        targets: {},
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'maybe', rationale: 'Somewhat unclear' }],
        responseAssessmentsByName: {
          safety: [{ name: 'safety', stringValue: 'yes', rationale: 'Somewhat fails criteria' }],
          customAssessment: [{ name: 'customAssessment', stringValue: 'no', rationale: 'Not clear' }],
        },
        metrics: {},
      },
      otherRunValue: {
        evaluationId: `eval-03`,
        requestId: `req-03`,
        inputs: {
          messages: [
            { role: 'user', content: 'Hello again' },
            { role: 'assistant', content: 'Hello again Response' },
          ],
        },
        inputsId: `inputs-03`,
        outputs: {},
        targets: {},
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes', rationale: 'Clear' }],
        responseAssessmentsByName: {
          safety: [{ name: 'safety', stringValue: 'no', rationale: 'Passes criteria' }],
          customAssessment: [{ name: 'customAssessment', stringValue: 'yes', rationale: 'Clear' }],
        },
        metrics: {},
      },
    },
  ];
  it('filter on search query matches deeply on inputs', () => {
    const assessmentFilters: AssessmentFilter[] = [];
    const searchQuery = 'hello again';

    const filteredResults = filterEvaluationResults(evals, assessmentFilters, searchQuery);

    expect(filteredResults).toEqual([evals[1]]);
  });
});

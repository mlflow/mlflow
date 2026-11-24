import { describe, it, expect } from '@jest/globals';

import { filterEvaluationResults } from './EvaluationsFilterUtils';
import { FilterOperator } from '../types';
import type { AssessmentFilter, EvalTraceComparisonEntry, RunEvaluationTracesDataEntry } from '../types';

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

  it('filter on search query matches in trace request preview', () => {
    const evalsWithTraceInfo: EvalTraceComparisonEntry[] = [
      {
        currentRunValue: {
          ...evals[0].currentRunValue!,
          inputs: { inputs: 'paris' },
        },
      },
      {
        currentRunValue: {
          ...evals[1].currentRunValue!,
          inputs: { inputs: 'london' },
        },
      },
    ];

    let filteredResults = filterEvaluationResults(evalsWithTraceInfo, [], 'paris');

    expect(filteredResults).toEqual([evalsWithTraceInfo[0]]);

    filteredResults = filterEvaluationResults(evalsWithTraceInfo, [], 'inputs');

    expect(filteredResults).toEqual(evalsWithTraceInfo);
  });

  it('filters on assessment value when multiple assessment value types exist for same name', () => {
    const makeEntry = (
      assessments: RunEvaluationTracesDataEntry['responseAssessmentsByName'],
    ): EvalTraceComparisonEntry => ({
      currentRunValue: {
        evaluationId: 'eval-1',
        requestId: 'req-1',
        inputs: {},
        inputsId: 'inputs-1',
        outputs: {},
        targets: {},
        overallAssessments: [],
        responseAssessmentsByName: assessments,
        metrics: {},
      },
    });

    const evalsWithMultipleAssessments: EvalTraceComparisonEntry[] = [
      makeEntry({
        mixedAssessment: [
          { name: 'mixedAssessment', errorMessage: 'Some error' },
          { name: 'mixedAssessment', stringValue: 'yes' },
        ],
      }),
      makeEntry({
        mixedAssessment: [{ name: 'mixedAssessment', stringValue: 'no' }],
      }),
      makeEntry({
        mixedAssessment: [{ name: 'mixedAssessment', errorMessage: 'Only error' }],
      }),
    ];

    const yesFilter: AssessmentFilter[] = [
      { assessmentName: 'mixedAssessment', filterValue: 'yes', run: 'currentRun' },
    ];
    const noFilter: AssessmentFilter[] = [{ assessmentName: 'mixedAssessment', filterValue: 'no', run: 'currentRun' }];

    const yesResults = filterEvaluationResults(evalsWithMultipleAssessments, yesFilter, undefined, 'currentRun');
    expect(yesResults).toHaveLength(1);
    expect(yesResults[0]).toBe(evalsWithMultipleAssessments[0]);

    const noResults = filterEvaluationResults(evalsWithMultipleAssessments, noFilter, undefined, 'currentRun');
    expect(noResults).toHaveLength(1);
    expect(noResults[0]).toBe(evalsWithMultipleAssessments[1]);
  });

  it.each([
    {
      description: '= matches string "1" against number 1',
      operator: undefined,
      expectedIndices: [0],
    },
    {
      description: '!= excludes string "1" when filtering against number 1',
      operator: FilterOperator.NOT_EQUALS,
      expectedIndices: [1, 2],
    },
  ])('filters numeric string values: $description', ({ operator, expectedIndices }) => {
    const makeEntry = (
      assessments: RunEvaluationTracesDataEntry['responseAssessmentsByName'],
    ): EvalTraceComparisonEntry => ({
      currentRunValue: {
        evaluationId: 'eval-1',
        requestId: 'req-1',
        inputs: {},
        inputsId: 'inputs-1',
        outputs: {},
        targets: {},
        overallAssessments: [],
        responseAssessmentsByName: assessments,
        metrics: {},
      },
    });

    const evalsWithNumericStrings: EvalTraceComparisonEntry[] = [
      makeEntry({ score: [{ name: 'score', stringValue: '1' }] }),
      makeEntry({ score: [{ name: 'score', stringValue: '2' }] }),
      makeEntry({ score: [{ name: 'score', stringValue: '3' }] }),
    ];

    const filter: AssessmentFilter[] = [
      { assessmentName: 'score', filterValue: 1, filterOperator: operator, run: 'currentRun' },
    ];

    const results = filterEvaluationResults(evalsWithNumericStrings, filter, undefined, 'currentRun');
    expect(results).toHaveLength(expectedIndices.length);
    expectedIndices.forEach((idx, i) => {
      expect(results[i]).toBe(evalsWithNumericStrings[idx]);
    });
  });

  it('filters on Error value to find assessments with errors', () => {
    const makeEntry = (
      assessments: RunEvaluationTracesDataEntry['responseAssessmentsByName'],
    ): EvalTraceComparisonEntry => ({
      currentRunValue: {
        evaluationId: 'eval-1',
        requestId: 'req-1',
        inputs: {},
        inputsId: 'inputs-1',
        outputs: {},
        targets: {},
        overallAssessments: [],
        responseAssessmentsByName: assessments,
        metrics: {},
      },
    });

    const evalsWithErrors: EvalTraceComparisonEntry[] = [
      makeEntry({
        testAssessment: [{ name: 'testAssessment', stringValue: 'yes' }],
      }),
      makeEntry({
        testAssessment: [{ name: 'testAssessment', errorMessage: 'Some error occurred' }],
      }),
      makeEntry({
        testAssessment: [{ name: 'testAssessment', stringValue: 'no' }],
      }),
    ];

    const errorFilter: AssessmentFilter[] = [
      { assessmentName: 'testAssessment', filterValue: 'Error', run: 'currentRun' },
    ];

    const errorResults = filterEvaluationResults(evalsWithErrors, errorFilter, undefined, 'currentRun');
    expect(errorResults).toHaveLength(1);
    expect(errorResults[0]).toBe(evalsWithErrors[1]);
  });

  it.each([
    {
      description: '= undefined returns traces with no assessments',
      operator: undefined,
      expectedIndices: [1],
    },
    {
      description: '!= undefined returns traces that have assessments',
      operator: FilterOperator.NOT_EQUALS,
      expectedIndices: [0, 2],
    },
  ])('filters with undefined filterValue: $description', ({ operator, expectedIndices }) => {
    const makeEntry = (
      assessments: RunEvaluationTracesDataEntry['responseAssessmentsByName'],
    ): EvalTraceComparisonEntry => ({
      currentRunValue: {
        evaluationId: 'eval-1',
        requestId: 'req-1',
        inputs: {},
        inputsId: 'inputs-1',
        outputs: {},
        targets: {},
        overallAssessments: [],
        responseAssessmentsByName: assessments,
        metrics: {},
      },
    });

    const evalsWithMixedAssessments: EvalTraceComparisonEntry[] = [
      makeEntry({ score: [{ name: 'score', stringValue: 'yes' }] }),
      makeEntry({}),
      makeEntry({ score: [{ name: 'score', stringValue: 'no' }] }),
    ];

    const filter: AssessmentFilter[] = [
      { assessmentName: 'score', filterValue: undefined, filterOperator: operator, run: 'currentRun' },
    ];

    const results = filterEvaluationResults(evalsWithMixedAssessments, filter, undefined, 'currentRun');
    expect(results).toHaveLength(expectedIndices.length);
    expectedIndices.forEach((idx, i) => {
      expect(results[i]).toBe(evalsWithMixedAssessments[idx]);
    });
  });

  it('filter on search query matches trace evaluationId (trace_id)', () => {
    const evalsWithTraceId: EvalTraceComparisonEntry[] = [
      {
        currentRunValue: {
          evaluationId: '11301f0bdf2dfa5a762a4bac74b45db1',
          requestId: 'req-1',
          inputs: { input: 'some input' },
          inputsId: '11301f0bdf2dfa5a762a4bac74b45db1',
          outputs: {},
          targets: {},
          overallAssessments: [],
          responseAssessmentsByName: {},
          metrics: {},
        },
      },
      {
        currentRunValue: {
          evaluationId: 'aabbccddaabbccddaabbccddaabbccdd',
          requestId: 'req-2',
          inputs: { input: 'other input' },
          inputsId: 'aabbccddaabbccddaabbccddaabbccdd',
          outputs: {},
          targets: {},
          overallAssessments: [],
          responseAssessmentsByName: {},
          metrics: {},
        },
      },
    ];

    // Search by full backend trace ID
    const results = filterEvaluationResults(evalsWithTraceId, [], '11301f0bdf2dfa5a762a4bac74b45db1');
    expect(results).toHaveLength(1);
    expect(results[0]).toBe(evalsWithTraceId[0]);

    // Search by partial backend trace ID
    const partialResults = filterEvaluationResults(evalsWithTraceId, [], '11301f0b');
    expect(partialResults).toHaveLength(1);
    expect(partialResults[0]).toBe(evalsWithTraceId[0]);
  });

  it('filter on search query matches fullTraceId (V4 trace ID format)', () => {
    const evalsWithFullTraceId: EvalTraceComparisonEntry[] = [
      {
        currentRunValue: {
          evaluationId: '11301f0bdf2dfa5a762a4bac74b45db1',
          requestId: 'req-1',
          inputs: { input: 'some input' },
          inputsId: '11301f0bdf2dfa5a762a4bac74b45db1',
          outputs: {},
          targets: {},
          overallAssessments: [],
          responseAssessmentsByName: {},
          metrics: {},
          fullTraceId: 'trace:/catalog.schema/11301f0bdf2dfa5a762a4bac74b45db1',
        },
      },
      {
        currentRunValue: {
          evaluationId: 'aabbccddaabbccddaabbccddaabbccdd',
          requestId: 'req-2',
          inputs: { input: 'other input' },
          inputsId: 'aabbccddaabbccddaabbccddaabbccdd',
          outputs: {},
          targets: {},
          overallAssessments: [],
          responseAssessmentsByName: {},
          metrics: {},
          fullTraceId: 'trace:/other.location/aabbccddaabbccddaabbccddaabbccdd',
        },
      },
    ];

    // Search by full V4 trace ID
    const results = filterEvaluationResults(
      evalsWithFullTraceId,
      [],
      'trace:/catalog.schema/11301f0bdf2dfa5a762a4bac74b45db1',
    );
    expect(results).toHaveLength(1);
    expect(results[0]).toBe(evalsWithFullTraceId[0]);

    // Search by catalog.schema (location part)
    const locationResults = filterEvaluationResults(evalsWithFullTraceId, [], 'catalog.schema');
    expect(locationResults).toHaveLength(1);
    expect(locationResults[0]).toBe(evalsWithFullTraceId[0]);
  });
});

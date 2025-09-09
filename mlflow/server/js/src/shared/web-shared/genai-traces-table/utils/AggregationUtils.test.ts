import type { ThemeType } from '@databricks/design-system';
import { I18nUtils } from '@databricks/i18n';

import {
  getAssessmentAggregateOverallFraction,
  getAssessmentInfos,
  getBarChartData,
  getUniqueValueCountsBySourceId,
} from './AggregationUtils';
import type {
  AssessmentAggregates,
  AssessmentDType,
  AssessmentInfo,
  AssessmentRunCounts,
  AssessmentValueType,
  RunEvaluationResultAssessment,
  RunEvaluationResultAssessmentSource,
  RunEvaluationTracesDataEntry,
  RunEvaluationTracesRetrievalChunk,
} from '../types';

describe('getAssessmentInfos', () => {
  function makeTracesFromAssessments(
    assessmentData: {
      overallAssessments?: RunEvaluationResultAssessment[];
      responseAssessmentsByName?: Record<string, RunEvaluationResultAssessment[]>;
      retrievalChunks?: RunEvaluationTracesRetrievalChunk[];
    }[],
  ): RunEvaluationTracesDataEntry[] {
    return assessmentData.map((entry, index) => ({
      evaluationId: `eval-${index}`,
      requestId: `req-${index}`,
      inputs: {},
      inputsId: `inputs-${index}`,
      outputs: {},
      targets: {},
      overallAssessments: entry.overallAssessments || [],
      responseAssessmentsByName: entry.responseAssessmentsByName || {},
      metrics: {},
      retrievalChunks: entry.retrievalChunks,
    }));
  }

  const intl = I18nUtils.createIntlWithLocale();

  it('should correctly determine assessment data types across multiple rows', () => {
    const currentEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes', rationale: 'Meets criteria' }],
        responseAssessmentsByName: {
          numericAssessment: [{ name: 'numericAssessment', numericValue: 0.85 }],
          boolAssessment: [{ name: 'boolAssessment', booleanValue: true }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no', rationale: 'Fails criteria' }],
        responseAssessmentsByName: {
          stringAssessment: [{ name: 'stringAssessment', stringValue: 'partial' }],
        },
      },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, undefined);

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'overall_assessment',
          dtype: 'pass-fail',
          uniqueValues: new Set(['yes', 'no']),
        }),
        expect.objectContaining({
          name: 'numericAssessment',
          dtype: 'numeric',
          uniqueValues: new Set([0.85]),
        }),
        expect.objectContaining({
          name: 'boolAssessment',
          dtype: 'boolean',
          uniqueValues: new Set([true]),
        }),
        expect.objectContaining({
          name: 'stringAssessment',
          dtype: 'string',
          uniqueValues: new Set(['partial']),
        }),
      ]),
    );
  });

  it('handles assessments with failures', () => {
    const currentEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes', rationale: 'Meets criteria' }],
        responseAssessmentsByName: {
          numericAssessment: [{ name: 'numericAssessment', numericValue: 0.85 }],
          errorAssessment: [{ name: 'errorAssessment', errorMessage: 'Error message' }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no', rationale: 'Fails criteria' }],
        responseAssessmentsByName: {
          numericAssessment: [{ name: 'numericAssessment', errorMessage: 'Error message' }],
          stringAssessment: [{ name: 'stringAssessment', stringValue: 'partial' }],
          errorAssessment: [{ name: 'errorAssessment', errorMessage: 'Error message again' }],
        },
      },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, undefined);

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'overall_assessment',
          dtype: 'pass-fail',
          uniqueValues: new Set(['yes', 'no']),
        }),
        expect.objectContaining({
          name: 'numericAssessment',
          dtype: 'numeric',
          uniqueValues: new Set([0.85]),
        }),
        expect.objectContaining({
          name: 'errorAssessment',
          dtype: 'unknown',
          uniqueValues: new Set(),
        }),
        expect.objectContaining({
          name: 'stringAssessment',
          dtype: 'string',
          uniqueValues: new Set(['partial', undefined]),
        }),
      ]),
    );
  });

  it('should correctly set isRetrievalAssessment if any rows has retrieval assessment', () => {
    const currentEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes', rationale: 'Meets criteria' }],
        responseAssessmentsByName: {
          numericAssessment: [{ name: 'numericAssessment', numericValue: 0.85 }],
          boolAssessment: [{ name: 'boolAssessment', booleanValue: true }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no', rationale: 'Fails criteria' }],
        responseAssessmentsByName: {
          stringAssessment: [{ name: 'stringAssessment', stringValue: 'partial' }],
        },
        retrievalChunks: [
          {
            content: 'retrieval content',
            docUrl: 'retrieval-doc-url',
            retrievalAssessmentsByName: { chunk_relevance: [{ name: 'chunk_relevance', stringValue: 'yes' }] },
          },
        ],
      },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, undefined);

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'overall_assessment',
          dtype: 'pass-fail',
          uniqueValues: new Set(['yes', 'no']),
        }),
        expect.objectContaining({
          name: 'numericAssessment',
          dtype: 'numeric',
          uniqueValues: new Set([0.85]),
        }),
        expect.objectContaining({
          name: 'boolAssessment',
          dtype: 'boolean',
          uniqueValues: new Set([true]),
        }),
        expect.objectContaining({
          name: 'stringAssessment',
          dtype: 'string',
          uniqueValues: new Set(['partial']),
        }),
        expect.objectContaining({
          name: 'chunk_relevance',
          dtype: 'pass-fail',
          isRetrievalAssessment: true,
          uniqueValues: new Set(['yes']),
        }),
      ]),
    );
  });

  it('should determine assessment data type as string for a mix of numeric and boolean rows', () => {
    const currentEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes' }],
        responseAssessmentsByName: {
          sharedAssessment: [{ name: 'sharedAssessment', booleanValue: true }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes' }],
        responseAssessmentsByName: {
          sharedAssessment: [{ name: 'sharedAssessment', numericValue: 0.5 }],
        },
      },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, undefined);

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'overall_assessment',
          dtype: 'pass-fail',
        }),
        expect.objectContaining({
          name: 'sharedAssessment',
          dtype: 'string',
        }),
      ]),
    );
  });

  it('should merge assessment values from multiple evaluations', () => {
    const currentEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes' }],
        responseAssessmentsByName: {
          numericAssessment: [{ name: 'numericAssessment', numericValue: 1.0 }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no' }],
        responseAssessmentsByName: {
          stringAssessment: [{ name: 'stringAssessment', stringValue: 'low' }],
        },
      },
    ]);

    const otherEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes' }],
        responseAssessmentsByName: {
          numericAssessment: [{ name: 'numericAssessment', numericValue: 0.75 }],
          boolAssessment: [{ name: 'boolAssessment', booleanValue: false }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no' }],
        responseAssessmentsByName: {
          stringAssessment: [{ name: 'stringAssessment', stringValue: 'high' }],
        },
      },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, otherEvaluationResults);

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'overall_assessment',
          dtype: 'pass-fail',
          uniqueValues: new Set(['yes', 'no']),
        }),
        expect.objectContaining({
          name: 'numericAssessment',
          dtype: 'numeric',
          uniqueValues: new Set([1.0, 0.75]),
        }),
        expect.objectContaining({
          name: 'boolAssessment',
          dtype: 'boolean',
          uniqueValues: new Set([false]),
        }),
        expect.objectContaining({
          name: 'stringAssessment',
          dtype: 'string',
          uniqueValues: new Set(['low', 'high']),
        }),
      ]),
    );
  });

  it('should exclude overall assessment if it has no valid values', () => {
    const currentEvaluationResults = makeTracesFromAssessments([
      { overallAssessments: [{ name: 'overall_assessment', stringValue: null }] },
      { overallAssessments: [{ name: 'overall_assessment', stringValue: null }] },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, undefined);
    expect(result).toEqual([]);
  });

  it('should mark assessments as editable when sourced from AI_JUDGE or HUMAN', () => {
    const aiSource: RunEvaluationResultAssessmentSource = {
      sourceType: 'AI_JUDGE',
      sourceId: 'ai-source-1',
      metadata: { model: 'GPT-4', version: '1.0' },
    };

    const humanSource: RunEvaluationResultAssessmentSource = {
      sourceType: 'HUMAN',
      sourceId: 'human-source-1',
      metadata: { reviewer: 'expert' },
    };

    const currentEvaluationResults = makeTracesFromAssessments([
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'yes', source: aiSource }],
        responseAssessmentsByName: {
          editableMetric: [{ name: 'editableMetric', stringValue: 'approved', source: humanSource }],
        },
      },
      {
        overallAssessments: [{ name: 'overall_assessment', stringValue: 'no', source: aiSource }],
        responseAssessmentsByName: {
          anotherEditableMetric: [{ name: 'anotherEditableMetric', booleanValue: true, source: aiSource }],
        },
      },
    ]);

    const result = getAssessmentInfos(intl, currentEvaluationResults, undefined);

    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'overall_assessment',
          isEditable: true,
        }),
        expect.objectContaining({
          name: 'editableMetric',
          isEditable: true,
        }),
        expect.objectContaining({
          name: 'anotherEditableMetric',
          isEditable: true,
        }),
      ]),
    );
  });
});

describe('getAssessmentAggregateOverallFraction', () => {
  it('should compute correct fraction for pass-fail dtype', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'overall_assessment',
      displayName: 'Overall Assessment',
      isKnown: true,
      isOverall: true,
      metricName: 'overall_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'pass-fail',
      uniqueValues: new Set(['yes', 'no']),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map([
      ['yes', 3],
      ['no', 2],
    ]);

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(3 / 5);
  });

  it('should return 0 when no pass values exist for pass-fail dtype', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'overall_assessment',
      displayName: 'Overall Assessment',
      isKnown: true,
      isOverall: true,
      metricName: 'overall_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'pass-fail',
      uniqueValues: new Set(['yes', 'no']),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map([['no', 4]]);

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(0);
  });

  it('should compute correct fraction for boolean dtype', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'bool_assessment',
      displayName: 'Boolean Assessment',
      isKnown: true,
      isOverall: false,
      metricName: 'bool_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'boolean',
      uniqueValues: new Set([true, false]),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map([
      [true, 7],
      [false, 3],
    ]);

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(7 / 10);
  });

  it('should return 0 when no true values exist for boolean dtype', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'bool_assessment',
      displayName: 'Boolean Assessment',
      isKnown: true,
      isOverall: false,
      metricName: 'bool_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'boolean',
      uniqueValues: new Set([true, false]),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map([[false, 5]]);

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(0);
  });

  it('should compute correct fraction for numeric dtype', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'bool_assessment',
      displayName: 'Boolean Assessment',
      isKnown: true,
      isOverall: false,
      metricName: 'bool_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'boolean',
      uniqueValues: new Set([true, false]),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map([
      [true, 7],
      [false, 3],
    ]);

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(7 / 10);
  });

  it('should return 0 for empty assessment counts', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'empty_assessment',
      displayName: 'Empty Assessment',
      isKnown: true,
      isOverall: false,
      metricName: 'empty_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'pass-fail',
      uniqueValues: new Set(['yes', 'no']),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map();

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(0);
  });

  it('should return 0 for unknown dtype', () => {
    const assessmentInfo: AssessmentInfo = {
      name: 'unknown_assessment',
      displayName: 'Unknown Assessment',
      isKnown: true,
      isOverall: false,
      metricName: 'unknown_assessment',
      source: undefined,
      isCustomMetric: false,
      isEditable: false,
      isRetrievalAssessment: false,
      dtype: 'string' as any, // invalid dtype to test default return
      uniqueValues: new Set(['yes', 'no']),
      docsLink: '',
      missingTooltip: '',
      description: '',
    };

    const assessmentRunCounts: AssessmentRunCounts = new Map([
      ['yes', 5],
      ['no', 5],
    ]);

    const result = getAssessmentAggregateOverallFraction(assessmentInfo, assessmentRunCounts);
    expect(result).toBe(0);
  });
});

describe('getBarChartData', () => {
  const intl = I18nUtils.createIntlWithLocale();
  const mockTheme = {
    colors: {
      textSecondary: 'red',
    },
  } as ThemeType;

  const createMockAssessmentInfo = (
    dtype: AssessmentDType,
    uniqueValues: AssessmentValueType[],
    containsErrors?: boolean,
  ): AssessmentInfo => ({
    name: 'assessment',
    displayName: 'Assessment',
    isKnown: true,
    isOverall: false,
    metricName: 'metric',
    source: undefined,
    isCustomMetric: false,
    isEditable: false,
    isRetrievalAssessment: false,
    dtype,
    uniqueValues: new Set(uniqueValues),
    docsLink: '',
    missingTooltip: '',
    description: '',
    containsErrors,
  });

  it('single run pass-fail', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('pass-fail', ['yes', 'no']);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        ['yes', 5],
        ['no', 3],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: 'Pass',
        current: expect.objectContaining({
          value: 5,
          fraction: 5 / 8,
          tooltip: '5/8 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
      expect.objectContaining({
        name: 'Fail',
        current: expect.objectContaining({
          value: 3,
          fraction: 3 / 8,
          tooltip: '3/8 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
    ]);
  });

  it('includes error bar chart', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('pass-fail', ['yes', 'no'], true);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        ['yes', 5],
        ['no', 3],
        ['Error', 2],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: 'Pass',
        current: expect.objectContaining({
          value: 5,
          fraction: 5 / 8,
          tooltip: '5/8 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
      expect.objectContaining({
        name: 'Fail',
        current: expect.objectContaining({
          value: 3,
          fraction: 3 / 8,
          tooltip: '3/8 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
      expect.objectContaining({
        name: 'Error',
        current: expect.objectContaining({
          value: 2,
          fraction: 0,
          tooltip: '2 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
    ]);
  });

  it('compare runs pass-fail', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('pass-fail', ['yes', 'no']);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        ['yes', 7],
        ['no', 3],
      ]),
      otherCounts: new Map([
        ['yes', 4],
        ['no', 6],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
      'Previous Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: 'Pass',
        current: expect.objectContaining({
          value: 7,
          fraction: 0.7,
          tooltip: '7/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 4,
          fraction: 0.4,
          tooltip: '4/10 for run "Previous Run"',
        }),
        scoreChange: 0.7 - 0.4,
      }),
      expect.objectContaining({
        name: 'Fail',
        current: expect.objectContaining({
          value: 3,
          fraction: 0.3,
          tooltip: '3/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 6,
          fraction: 0.6,
          tooltip: '6/10 for run "Previous Run"',
        }),
        scoreChange: 0.3 - 0.6,
      }),
    ]);
  });

  it('single run boolean', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('boolean', [true, false]);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        [true, 8],
        [false, 2],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: 'True',
        current: expect.objectContaining({
          value: 8,
          fraction: 0.8,
          tooltip: '8/10 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
      expect.objectContaining({
        name: 'False',
        current: expect.objectContaining({
          value: 2,
          fraction: 0.2,
          tooltip: '2/10 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
    ]);
  });

  it('compare runs boolean', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('boolean', [true, false]);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        [true, 6],
        [false, 4],
      ]),
      otherCounts: new Map([
        [true, 3],
        [false, 7],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
      'Previous Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: 'True',
        current: expect.objectContaining({
          value: 6,
          fraction: 0.6,
          tooltip: '6/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 3,
          fraction: 0.3,
          tooltip: '3/10 for run "Previous Run"',
        }),
        scoreChange: 0.6 - 0.3,
      }),
      expect.objectContaining({
        name: 'False',
        current: expect.objectContaining({
          value: 4,
          fraction: 0.4,
          tooltip: '4/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 7,
          fraction: 0.7,
          tooltip: '7/10 for run "Previous Run"',
        }),
        scoreChange: 0.4 - 0.7,
      }),
    ]);
  });

  it('single run numeric', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('numeric', [1, 2, 3]);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        [1, 4],
        [2, 3],
        [3, 3],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: '1',
        current: expect.objectContaining({
          value: 4,
          fraction: 0.4,
          tooltip: '4/10 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
      expect.objectContaining({
        name: '2',
        current: expect.objectContaining({
          value: 3,
          fraction: 0.3,
          tooltip: '3/10 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
      expect.objectContaining({
        name: '3',
        current: expect.objectContaining({
          value: 3,
          fraction: 0.3,
          tooltip: '3/10 for run "Current Run"',
        }),
        scoreChange: undefined,
      }),
    ]);
  });

  it('compare runs numeric', () => {
    const mockAssessmentInfo = createMockAssessmentInfo('numeric', [1, 2, 3]);
    const displayInfoCounts: AssessmentAggregates = {
      assessmentInfo: mockAssessmentInfo,
      currentCounts: new Map([
        [1, 6],
        [2, 2],
        [3, 2],
      ]),
      otherCounts: new Map([
        [1, 3],
        [2, 4],
        [3, 3],
      ]),
      currentNumRootCause: 0,
      otherNumRootCause: 0,
      assessmentFilters: [],
    };

    const result = getBarChartData(
      intl,
      mockTheme,
      mockAssessmentInfo,
      [],
      jest.fn(),
      displayInfoCounts,
      'Current Run',
      'Previous Run',
    );

    expect(result).toEqual([
      expect.objectContaining({
        name: '1',
        current: expect.objectContaining({
          value: 6,
          fraction: 0.6,
          tooltip: '6/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 3,
          fraction: 0.3,
          tooltip: '3/10 for run "Previous Run"',
        }),
        scoreChange: 0.6 - 0.3,
      }),
      expect.objectContaining({
        name: '2',
        current: expect.objectContaining({
          value: 2,
          fraction: 0.2,
          tooltip: '2/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 4,
          fraction: 0.4,
          tooltip: '4/10 for run "Previous Run"',
        }),
        scoreChange: 0.2 - 0.4,
      }),
      expect.objectContaining({
        name: '3',
        current: expect.objectContaining({
          value: 2,
          fraction: 0.2,
          tooltip: '2/10 for run "Current Run"',
        }),
        other: expect.objectContaining({
          value: 3,
          fraction: 0.3,
          tooltip: '3/10 for run "Previous Run"',
        }),
        scoreChange: 0.2 - 0.3,
      }),
    ]);
  });
});

describe('getUniqueValueCountsBySourceId', () => {
  const mockAssessmentInfo: AssessmentInfo = {
    name: 'test_assessment',
    displayName: 'Test Assessment',
    isKnown: true,
    isOverall: false,
    metricName: 'test_metric',
    isCustomMetric: false,
    isEditable: false,
    isRetrievalAssessment: false,
    dtype: 'string',
    uniqueValues: new Set(['true', 'false']),
    docsLink: '',
    missingTooltip: '',
    description: '',
  };

  it('should return count {false: 1}', () => {
    const assessments: RunEvaluationResultAssessment[] = [
      {
        name: 'test_assessment',
        stringValue: 'false',
        source: { sourceType: 'AI_JUDGE', sourceId: 'databricks', metadata: {} },
      },
    ];

    expect(getUniqueValueCountsBySourceId(mockAssessmentInfo, assessments)).toEqual([
      { value: 'false', count: 1, latestAssessment: assessments[0] },
    ]);
  });

  it('should include error counts', () => {
    const assessments: RunEvaluationResultAssessment[] = [
      {
        name: 'test_assessment',
        errorMessage: 'Error message',
        source: { sourceType: 'AI_JUDGE', sourceId: 'databricks', metadata: {} },
      },
    ];

    expect(getUniqueValueCountsBySourceId(mockAssessmentInfo, assessments)).toEqual([
      { value: 'Error', count: 1, latestAssessment: assessments[0] },
    ]);
  });

  it('should return all counts when different AI values are provided', () => {
    const assessments: RunEvaluationResultAssessment[] = [
      {
        name: 'test_assessment',
        stringValue: 'false',
        source: { sourceType: 'AI_JUDGE', sourceId: 'databricks', metadata: {} },
        timestamp: 200,
      },
      {
        name: 'test_assessment',
        stringValue: 'true',
        source: { sourceType: 'AI_JUDGE', sourceId: 'databricks', metadata: {} },
        timestamp: 100,
      },
    ];

    expect(getUniqueValueCountsBySourceId(mockAssessmentInfo, assessments)).toEqual([
      { value: 'false', count: 1, latestAssessment: assessments[0] },
      { value: 'true', count: 1, latestAssessment: assessments[1] },
    ]);
  });

  it('should prioritize HUMAN and AI_JUDGE equally', () => {
    const assessments: RunEvaluationResultAssessment[] = [
      {
        name: 'test_assessment',
        stringValue: 'true',
        source: { sourceType: 'HUMAN', sourceId: 'user1', metadata: {} },
      },
      {
        name: 'test_assessment',
        stringValue: 'false',
        source: { sourceType: 'AI_JUDGE', sourceId: 'databricks', metadata: {} },
      },
    ];

    expect(getUniqueValueCountsBySourceId(mockAssessmentInfo, assessments)).toEqual([
      {
        value: 'true',
        count: 1,
        latestAssessment: assessments[0],
      },
      {
        value: 'false',
        count: 1,
        latestAssessment: assessments[1],
      },
    ]);
  });

  it('should handle multiple human sources and AI judge correctly', () => {
    const assessments: RunEvaluationResultAssessment[] = [
      {
        name: 'test_assessment',
        stringValue: 'true',
        source: { sourceType: 'HUMAN', sourceId: 'user1', metadata: {} },
        timestamp: 400,
      },
      {
        name: 'test_assessment',
        stringValue: 'false',
        source: { sourceType: 'HUMAN', sourceId: 'user1', metadata: {} },
        timestamp: 300,
      },
      {
        name: 'test_assessment',
        stringValue: 'false',
        source: { sourceType: 'HUMAN', sourceId: 'user2', metadata: {} },
        timestamp: 200,
      },
      {
        name: 'test_assessment',
        stringValue: 'true',
        source: { sourceType: 'AI_JUDGE', sourceId: 'databricks', metadata: {} },
        timestamp: 100,
      },
    ];

    expect(getUniqueValueCountsBySourceId(mockAssessmentInfo, assessments)).toEqual([
      { value: 'true', count: 2, latestAssessment: assessments[0] },
      { value: 'false', count: 2, latestAssessment: assessments[1] },
    ]);
  });
});

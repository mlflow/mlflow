import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AssessmentCell } from './rendererFunctions';
import { GenAITracesTableContext } from '../GenAITracesTableContext';
import { ModelTraceExplorerRunJudgesContextProvider } from '../../model-trace-explorer/contexts/RunJudgesContext';
import { createTestAssessmentInfo, createTestTraceInfoV3 } from '../test-fixtures/EvaluatedTraceTestUtils';
import type { EvalTraceComparisonEntry } from '../types';
import { applyTraceInfoV3ToEvalEntry } from '../utils/TraceUtils';

jest.mock('../../model-trace-explorer/FeatureUtils', () => ({
  shouldUseUnifiedModelTraceComparisonUI: () => false,
  isEvaluatingTracesInDetailsViewEnabled: () => true,
}));

const TRACE_ID = 'trace-abc';
const JUDGE_NAME = 'Safety';

const makeComparisonEntry = (traceId: string): EvalTraceComparisonEntry => {
  const traceInfo = createTestTraceInfoV3(traceId, 'req-1', 'Hello', [], 'exp-1');
  const [entry] = applyTraceInfoV3ToEvalEntry([
    {
      evaluationId: traceId,
      requestId: 'req-1',
      inputsId: traceId,
      inputs: {},
      outputs: {},
      targets: {},
      overallAssessments: [],
      responseAssessmentsByName: {},
      metrics: {},
      traceInfo,
    },
  ]);
  return { currentRunValue: entry };
};

const renderCell = (
  traceId: string,
  assessmentName: string,
  evaluations: React.ComponentProps<typeof ModelTraceExplorerRunJudgesContextProvider>['evaluations'] = {},
) => {
  const assessmentInfo = createTestAssessmentInfo(assessmentName, assessmentName, 'pass-fail');
  const comparisonEntry = makeComparisonEntry(traceId);

  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ModelTraceExplorerRunJudgesContextProvider evaluations={evaluations}>
          <GenAITracesTableContext.Provider value={{ isGroupedBySession: false } as any}>
            <AssessmentCell isComparing={false} assessmentInfo={assessmentInfo} comparisonEntry={comparisonEntry} />
          </GenAITracesTableContext.Provider>
        </ModelTraceExplorerRunJudgesContextProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

// The Databricks Spinner renders with this class
const querySpinner = () => document.querySelector('.du-bois-light-spin');

describe('AssessmentCell — judge running spinner', () => {
  it('does not show a spinner when no evaluation is running', () => {
    renderCell(TRACE_ID, JUDGE_NAME, {});
    expect(querySpinner()).not.toBeInTheDocument();
  });

  it('shows a spinner when a matching judge is loading for this trace and column', () => {
    renderCell(TRACE_ID, JUDGE_NAME, {
      'eval-key-1': {
        requestKey: 'eval-key-1',
        label: JUDGE_NAME,
        isLoading: true,
        tracesData: { [TRACE_ID]: {} as any },
      },
    });

    expect(querySpinner()).toBeInTheDocument();
  });

  it('does NOT show a spinner when a different judge (different label) is loading', () => {
    renderCell(TRACE_ID, 'demo', {
      'eval-key-1': {
        requestKey: 'eval-key-1',
        label: JUDGE_NAME, // "Safety" — different from the "demo" column
        isLoading: true,
        tracesData: { [TRACE_ID]: {} as any },
      },
    });

    expect(querySpinner()).not.toBeInTheDocument();
  });

  it('does NOT show a spinner when the trace ID is not in the loading evaluation', () => {
    renderCell(TRACE_ID, JUDGE_NAME, {
      'eval-key-1': {
        requestKey: 'eval-key-1',
        label: JUDGE_NAME,
        isLoading: true,
        tracesData: { 'some-other-trace': {} as any }, // different trace
      },
    });

    expect(querySpinner()).not.toBeInTheDocument();
  });

  it('does NOT show a spinner when the evaluation has finished (isLoading = false)', () => {
    renderCell(TRACE_ID, JUDGE_NAME, {
      'eval-key-1': {
        requestKey: 'eval-key-1',
        label: JUDGE_NAME,
        isLoading: false,
        tracesData: { [TRACE_ID]: {} as any },
      },
    });

    expect(querySpinner()).not.toBeInTheDocument();
  });
});

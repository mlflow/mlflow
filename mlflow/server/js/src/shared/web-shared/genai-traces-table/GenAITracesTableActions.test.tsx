import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { GenAITracesTableActions } from './GenAITracesTableActions';
import { GenAITracesTableContext } from './GenAITracesTableContext';
import type { TraceActions } from './types';
import { createTestTraceInfoV3 } from './test-fixtures/EvaluatedTraceTestUtils';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';
import type { RunEvaluationTracesDataEntry } from './types';

jest.mock('./utils/FeatureUtils', () => ({
  shouldEnableTagGrouping: () => false,
}));

jest.mock('../model-trace-explorer/FeatureUtils', () => ({
  shouldUseUnifiedModelTraceComparisonUI: () => false,
}));

const EXPERIMENT_ID = 'test-experiment-id';

const makeSelectedTraces = (traceInfos: ModelTraceInfoV3[]): RunEvaluationTracesDataEntry[] =>
  traceInfos.map((t) => ({
    evaluationId: t.trace_id,
    requestId: t.client_request_id ?? t.trace_id,
    inputsId: t.trace_id,
    inputs: {},
    outputs: {},
    targets: {},
    overallAssessments: [],
    responseAssessmentsByName: {},
    metrics: {},
    traceInfo: t,
  }));

const renderActions = (traceActions: TraceActions, traceInfos: ModelTraceInfoV3[] = []) => {
  const selectedTraces = makeSelectedTraces(traceInfos);
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        {/* Provide a minimal context so the component doesn't crash */}
        <GenAITracesTableContext.Provider
          value={{ table: undefined, selectedRowIds: [], isGroupedBySession: false } as any}
        >
          <GenAITracesTableActions
            experimentId={EXPERIMENT_ID}
            traceActions={traceActions}
            traceInfos={traceInfos}
            selectedTraces={selectedTraces}
          />
        </GenAITracesTableContext.Provider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

const openActionsDropdown = async () => {
  // The button label is "Actions (N)" when traces are selected
  const button = screen.getByRole('button', { name: /actions/i });
  await userEvent.click(button);
};

describe('GenAITracesTableActions — Run judges', () => {
  const traceInfos = [
    createTestTraceInfoV3('trace-1', 'req-1', 'Hello', [], EXPERIMENT_ID),
    createTestTraceInfoV3('trace-2', 'req-2', 'World', [], EXPERIMENT_ID),
  ];

  it('shows "Run judges" alongside "Add to evaluation dataset" when both actions are provided', async () => {
    renderActions(
      {
        exportToEvals: true,
        runJudgesAction: { showRunJudgesModal: jest.fn(), RunJudgesModal: null },
      },
      traceInfos,
    );

    await openActionsDropdown();

    expect(screen.getByRole('menuitem', { name: 'Run judges' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Add to evaluation dataset' })).toBeInTheDocument();
  });

  it('shows "Run judges" even when exportToEvals is false', async () => {
    renderActions(
      {
        exportToEvals: false,
        runJudgesAction: { showRunJudgesModal: jest.fn(), RunJudgesModal: null },
      },
      traceInfos,
    );

    await openActionsDropdown();

    expect(screen.getByRole('menuitem', { name: 'Run judges' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Add to evaluation dataset' })).not.toBeInTheDocument();
  });

  it('does not show "Run judges" when runJudgesAction is not provided', async () => {
    renderActions({ exportToEvals: true }, traceInfos);

    await openActionsDropdown();

    expect(screen.queryByRole('menuitem', { name: 'Run judges' })).not.toBeInTheDocument();
  });

  it('calls showRunJudgesModal with the selected trace IDs when clicked', async () => {
    const showRunJudgesModal = jest.fn();
    renderActions(
      {
        exportToEvals: true,
        runJudgesAction: { showRunJudgesModal, RunJudgesModal: null },
      },
      traceInfos,
    );

    await openActionsDropdown();
    await userEvent.click(screen.getByRole('menuitem', { name: 'Run judges' }));

    expect(showRunJudgesModal).toHaveBeenCalledTimes(1);
    expect(showRunJudgesModal).toHaveBeenCalledWith(['trace-1', 'trace-2']);
  });

  it('shows "Use for evaluation" group label when runJudgesAction is set without exportToEvals', async () => {
    renderActions(
      {
        exportToEvals: false,
        runJudgesAction: { showRunJudgesModal: jest.fn(), RunJudgesModal: null },
        deleteTracesAction: { deleteTraces: jest.fn<() => Promise<void>>().mockResolvedValue(undefined) },
      },
      traceInfos,
    );

    await openActionsDropdown();

    expect(screen.getByText('Use for evaluation')).toBeInTheDocument();
    expect(screen.getByText('Edit')).toBeInTheDocument();
  });
});

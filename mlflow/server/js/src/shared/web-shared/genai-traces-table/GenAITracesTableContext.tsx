import type { Table } from '@tanstack/react-table';
import { compact, isUndefined } from 'lodash';
import React, { createContext, useCallback, useMemo, useState } from 'react';

import type { GetTraceFunction } from './hooks/useGetTrace';
import type { EvalTraceComparisonEntry, RunEvaluationTracesDataEntry } from './types';
import type { ModelTraceInfoV3 } from '../model-trace-explorer';

type TraceRow = EvalTraceComparisonEntry & { multiline?: boolean };

export interface GenAITracesTableContextValue<T> {
  /** TanStack table instance (may be undefined until grandchild mounts) */
  table: Table<T> | undefined;
  /** Grandchild calls this once when it builds the table */
  setTable: (tbl: Table<T> | undefined) => void;

  selectedRowIds: string[];
  /** Grandchild updates this on every selection change */
  setSelectedRowIds: (rowIds: string[]) => void;

  /** Whether traces are grouped by session */
  isGroupedBySession: boolean;

  /**
   * Function to show the "Add to Evaluation Dataset" modal.
   * Provide traces to be added to the dataset. If `undefined` is passed, the modal is closed.
   */
  showAddToEvaluationDatasetModal?: (traces?: RunEvaluationTracesDataEntry[]) => void;

  /**
   * temporary hack to allow the trace drawer to render
   * the evaluation dataset modal in its own context.
   * `showAddToEvaluationDatasetModal` renders the modal
   * at the trace table root, but unfortunately this
   * makes it not a child of the trace drawer, causing
   * issues with interactivity.
   *
   * TODO: upgrade design system to use radix-modal to fix
   */
  renderExportTracesToDatasetsModal?: ({
    selectedTraceInfos,
    experimentId,
    visible,
    setVisible,
  }: {
    selectedTraceInfos: ModelTraceInfoV3[];
    experimentId: string;
    visible: boolean;
    setVisible: (visible: boolean) => void;
  }) => React.ReactNode;
}
export const GenAITracesTableContext = createContext<GenAITracesTableContextValue<TraceRow>>({
  table: undefined,
  setTable: () => {},
  selectedRowIds: [],
  setSelectedRowIds: () => {},
  isGroupedBySession: false,
  renderExportTracesToDatasetsModal: () => null,
});

interface GenAITracesTableProviderProps {
  children: React.ReactNode;
  experimentId?: string;
  getTrace?: GetTraceFunction;
  isGroupedBySession: boolean;

  /**
   * Provide a custom function to render the "Export Traces to Datasets" modal.
   */
  renderExportTracesToDatasetsModal?: ({
    selectedTraceInfos,
    experimentId,
    visible,
    setVisible,
  }: {
    selectedTraceInfos: ModelTraceInfoV3[];
    experimentId: string;
    visible: boolean;
    setVisible: (visible: boolean) => void;
  }) => React.ReactNode;
}

export const GenAITracesTableProvider: React.FC<React.PropsWithChildren<GenAITracesTableProviderProps>> = ({
  children,
  experimentId,
  getTrace,
  isGroupedBySession,
  renderExportTracesToDatasetsModal,
}) => {
  const [table, setTable] = useState<Table<TraceRow> | undefined>();
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [selectedTraces, setSelectedTraces] = useState<RunEvaluationTracesDataEntry[] | undefined>(undefined);

  const showAddToEvaluationDatasetModal = useCallback((traces?: RunEvaluationTracesDataEntry[]) => {
    setSelectedTraces(traces);
    setShowDatasetModal(!isUndefined(traces));
  }, []);

  const value = useMemo(
    () => ({
      table,
      setTable,
      selectedRowIds,
      setSelectedRowIds,
      isGroupedBySession,
      showAddToEvaluationDatasetModal,
      renderExportTracesToDatasetsModal,
    }),
    // prettier-ignore
    [
      table,
      selectedRowIds,
      isGroupedBySession,
      showAddToEvaluationDatasetModal,
      renderExportTracesToDatasetsModal,
    ],
  );

  return (
    // prettier-ignore
    <GenAITracesTableContext.Provider value={value}>
      {children}
      {renderExportTracesToDatasetsModal?.({
        selectedTraceInfos: selectedTraces ? compact(selectedTraces.map((trace) => trace.traceInfo)) : [],
        experimentId: experimentId ?? '',
        visible: showDatasetModal,
        setVisible: setShowDatasetModal,
      })}
    </GenAITracesTableContext.Provider>
  );
};

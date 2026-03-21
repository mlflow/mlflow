import type { Table } from '@tanstack/react-table';
import { compact, isUndefined } from 'lodash';
import React, { createContext, useCallback, useMemo, useState } from 'react';

import type { EvalTraceComparisonEntry, RunEvaluationTracesDataEntry } from './types';
import { ModelTraceExplorerPreferencesProvider } from '../model-trace-explorer/ModelTraceExplorerPreferencesContext';
import { useModelTraceExplorerContext } from '../model-trace-explorer/ModelTraceExplorerContext';
import type { GetTraceFunction } from './hooks/useGetTrace';
import { getExperimentIdFromTraceLocation } from './utils/TraceUtils';

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
}
export const GenAITracesTableContext = createContext<GenAITracesTableContextValue<TraceRow>>({
  table: undefined,
  setTable: () => {},
  selectedRowIds: [],
  isGroupedBySession: false,
  setSelectedRowIds: () => {},
});

interface GenAITracesTableProviderProps {
  children: React.ReactNode;
  experimentId?: string;
  getTrace?: GetTraceFunction;
  isGroupedBySession: boolean;
}

export const GenAITracesTableProvider: React.FC<React.PropsWithChildren<GenAITracesTableProviderProps>> = ({
  children,
  experimentId,
  getTrace,
  isGroupedBySession,
}) => {
  const [table, setTable] = useState<Table<TraceRow> | undefined>();
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [selectedTraces, setSelectedTraces] = useState<RunEvaluationTracesDataEntry[] | undefined>(undefined);

  const { renderExportTracesToDatasetsModal } = useModelTraceExplorerContext();

  const showAddToEvaluationDatasetModal = useCallback((traces?: RunEvaluationTracesDataEntry[]) => {
    setSelectedTraces(traces);
    setShowDatasetModal(!isUndefined(traces));
  }, []);

  const value = useMemo(
    () => ({
      table,
      setTable,
      getTrace,
      selectedRowIds,
      setSelectedRowIds,
      isGroupedBySession,
      showAddToEvaluationDatasetModal,
    }),
    [table, getTrace, selectedRowIds, isGroupedBySession, showAddToEvaluationDatasetModal],
  );

  return (
    <ModelTraceExplorerPreferencesProvider>
      <GenAITracesTableContext.Provider value={value}>
        {children}
        {renderExportTracesToDatasetsModal?.({
          selectedTraceInfos: selectedTraces ? compact(selectedTraces.map((trace) => trace.traceInfo)) : [],
          experimentId:
            getExperimentIdFromTraceLocation(selectedTraces?.[0]?.traceInfo?.trace_location) ?? experimentId ?? '',
          visible: showDatasetModal,
          setVisible: setShowDatasetModal,
        })}
      </GenAITracesTableContext.Provider>
    </ModelTraceExplorerPreferencesProvider>
  );
};

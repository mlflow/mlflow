import { useCallback, useEffect, useMemo, useState } from 'react';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import { isObject, sortBy } from 'lodash';
import { ExperimentViewTracesTableColumns } from '../TracesView.utils';

type LocalStorageStore = ReturnType<typeof LocalStorageUtils.getStoreForComponent>;

export interface ExperimentViewTracesUIState {
  hiddenColumns?: string[];
}

const defaultExperimentViewTracesUIState: ExperimentViewTracesUIState = {
  hiddenColumns: [ExperimentViewTracesTableColumns.traceName, ExperimentViewTracesTableColumns.source],
};

const loadExperimentViewTracesUIState = (localStore: LocalStorageStore): ExperimentViewTracesUIState => {
  try {
    const uiStateRaw = localStore.getItem('uiState');
    const uiState = JSON.parse(uiStateRaw);
    if (!isObject(uiState)) {
      return defaultExperimentViewTracesUIState;
    }
    return uiState;
  } catch (e) {
    return defaultExperimentViewTracesUIState;
  }
};

export const useExperimentViewTracesUIState = (experimentIds: string[]) => {
  const localStore = useMemo(() => {
    const persistenceIdentifier = JSON.stringify(experimentIds.slice().sort());
    return LocalStorageUtils.getStoreForComponent('ExperimentViewTraces', persistenceIdentifier);
  }, [experimentIds]);

  const [uiState, setUIState] = useState<ExperimentViewTracesUIState>(() =>
    loadExperimentViewTracesUIState(localStore),
  );

  const toggleHiddenColumn = useCallback((columnId: string) => {
    setUIState((prevUIState) => {
      const hiddenColumns = prevUIState.hiddenColumns || [];
      return {
        hiddenColumns: hiddenColumns.includes(columnId)
          ? hiddenColumns.filter((id) => id !== columnId)
          : [...hiddenColumns, columnId],
      };
    });
  }, []);

  useEffect(() => {
    localStore.setItem('uiState', JSON.stringify(uiState));
  }, [localStore, uiState]);

  return { uiState, toggleHiddenColumn };
};

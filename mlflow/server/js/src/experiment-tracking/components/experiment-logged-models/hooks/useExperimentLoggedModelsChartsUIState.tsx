import { useCallback, useEffect, useReducer, useState } from 'react';
import type { ExperimentRunsChartsUIConfiguration } from '../../experiment-page/models/ExperimentPageUIState';
import type { ChartSectionConfig } from '../../../types';
import type {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartsMetricByDatasetEntry,
} from '../../runs-charts/runs-charts.types';
import { RunsChartType } from '../../runs-charts/runs-charts.types';
import { isEmpty, uniq } from 'lodash';
import type { RunsChartsUIConfigurationSetter } from '../../runs-charts/hooks/useRunsChartsUIConfiguration';

type UpdateChartStateAction = { type: 'UPDATE'; stateSetter: RunsChartsUIConfigurationSetter };
type InitializeChartStateAction = { type: 'INITIALIZE'; initialConfig?: LoggedModelsChartsUIConfiguration };
type NewLoggedModelsStateAction = { type: 'METRICS_UPDATED'; metricsByDatasets: RunsChartsMetricByDatasetEntry[] };

type ChartsReducerAction = UpdateChartStateAction | NewLoggedModelsStateAction | InitializeChartStateAction;

interface LoggedModelsChartsUIConfiguration extends ExperimentRunsChartsUIConfiguration {
  isDirty: boolean;
}

const createLocalStorageKey = (storeIdentifier: string, version = 1) =>
  `experiment-logged-models-charts-ui-state-v${version}-${storeIdentifier}`;

/**
 * Generates a list of chart tiles based on logged models metrics and datasets.
 */
const getExperimentLoggedModelsPageChartSetup = (metricsByDatasets: RunsChartsMetricByDatasetEntry[]) => {
  const compareRunCharts: RunsChartsBarCardConfig[] = metricsByDatasets.map(
    ({ dataAccessKey, metricKey, datasetName }) => ({
      deleted: false,
      type: RunsChartType.BAR,
      uuid: `autogen-${dataAccessKey}`,
      metricSectionId: datasetName ? `autogen-${datasetName}` : 'default',
      isGenerated: true,
      metricKey,
      dataAccessKey,
      datasetName,
      displayName: datasetName ? `(${datasetName}) ${metricKey}` : undefined,
    }),
  );

  const compareRunSections: ChartSectionConfig[] = uniq(metricsByDatasets.map(({ datasetName }) => datasetName)).map(
    (datasetName) => ({
      display: true,
      name: datasetName ?? 'Metrics',
      uuid: datasetName ? `autogen-${datasetName}` : 'default',
      isReordered: false,
    }),
  );

  if (isEmpty(compareRunSections)) {
    compareRunSections.push({
      display: true,
      name: 'Metrics',
      uuid: 'default',
      isReordered: false,
    });
  }

  return {
    compareRunCharts,
    compareRunSections,
  };
};

// Internal utility function  used to merge the current charts state with potentially incoming new charts and sections
const reconcileChartsAndSections = (
  currentState: LoggedModelsChartsUIConfiguration,
  newCharts: { compareRunCharts: RunsChartsCardConfig[]; compareRunSections: ChartSectionConfig[] },
) => {
  // If there are no charts / sections, or if the state is in pristine state, just set the new charts if they're not empty
  if (!currentState.compareRunCharts || !currentState.compareRunSections || !currentState.isDirty) {
    if (newCharts.compareRunCharts.length > 0 || newCharts.compareRunSections.length > 0) {
      return {
        ...currentState,
        compareRunCharts: newCharts.compareRunCharts ?? [],
        compareRunSections: newCharts.compareRunSections ?? [],
      };
    }
  }

  // Otherwise, detect new sections and charts and add them to the list
  const newChartsToAdd = newCharts.compareRunCharts.filter(
    (newChart) => !currentState.compareRunCharts?.find((chart) => chart.uuid === newChart.uuid),
  );
  const newSectionsToAdd = newCharts.compareRunSections.filter(
    (newSection) =>
      newChartsToAdd.find((newChart) => newChart.metricSectionId === newSection.uuid) &&
      !currentState.compareRunSections?.find((section) => section.uuid === newSection.uuid),
  );

  if (newSectionsToAdd.length > 0 || newChartsToAdd.length > 0) {
    return {
      ...currentState,
      compareRunCharts: currentState.compareRunCharts
        ? [...currentState.compareRunCharts, ...newChartsToAdd]
        : newCharts.compareRunCharts,
      compareRunSections: currentState.compareRunSections
        ? [...currentState.compareRunSections, ...newSectionsToAdd]
        : newCharts.compareRunSections,
    };
  }
  return currentState;
};

const chartsUIStateInitializer = (): LoggedModelsChartsUIConfiguration => ({
  compareRunCharts: undefined,
  compareRunSections: undefined,
  autoRefreshEnabled: false,
  isAccordionReordered: false,
  chartsSearchFilter: '',
  globalLineChartConfig: undefined,
  isDirty: false,
});

// Reducer to manage the state of the charts UI
const chartsUIStateReducer = (state: LoggedModelsChartsUIConfiguration, action: ChartsReducerAction) => {
  // 'UPDATE' is sent by controls that updates the UI state directly
  if (action.type === 'UPDATE') {
    return { ...action.stateSetter(state), isDirty: true };
  }
  // 'METRICS_UPDATED' is sent when new logged models data is available and potentially new charts need to be added
  if (action.type === 'METRICS_UPDATED') {
    const { compareRunCharts, compareRunSections } = getExperimentLoggedModelsPageChartSetup(action.metricsByDatasets);
    const newState = reconcileChartsAndSections(state, { compareRunCharts, compareRunSections });
    return newState;
  }
  if (action.type === 'INITIALIZE') {
    if (action.initialConfig) {
      return action.initialConfig;
    }
  }
  return state;
};

const loadPersistedDataFromStorage = async (storeIdentifier: string) => {
  // This function is async on purpose to accommodate potential asynchoronous storage mechanisms (e.g. IndexedDB) in the future
  const serializedData = localStorage.getItem(createLocalStorageKey(storeIdentifier));
  if (!serializedData) {
    return undefined;
  }
  try {
    return JSON.parse(serializedData);
  } catch {
    return undefined;
  }
};

const saveDataToStorage = async (storeIdentifier: string, dataToPersist: LoggedModelsChartsUIConfiguration) => {
  localStorage.setItem(createLocalStorageKey(storeIdentifier), JSON.stringify(dataToPersist));
};

export const useExperimentLoggedModelsChartsUIState = (
  metricsByDatasets: RunsChartsMetricByDatasetEntry[],
  storeIdentifier: string,
) => {
  const [chartUIState, dispatchChartUIState] = useReducer(chartsUIStateReducer, undefined, chartsUIStateInitializer);
  const [loading, setLoading] = useState(true);

  // Attempt to load the persisted data when the component mounts
  useEffect(() => {
    setLoading(true);
    loadPersistedDataFromStorage(storeIdentifier).then((data) => {
      dispatchChartUIState({ type: 'INITIALIZE', initialConfig: data });
      setLoading(false);
    });
  }, [storeIdentifier]);

  // Attempt to update the charts state when the logged models change
  useEffect(() => {
    if (loading) {
      return;
    }
    dispatchChartUIState({ type: 'METRICS_UPDATED', metricsByDatasets });
  }, [metricsByDatasets, loading]);

  // Attempt persist the data when the state changes
  useEffect(() => {
    if (chartUIState.isDirty) {
      saveDataToStorage(storeIdentifier, chartUIState);
    }
  }, [storeIdentifier, chartUIState]);

  // Create an updater function to pass it to chart controls
  const updateUIState = useCallback(
    (stateSetter: RunsChartsUIConfigurationSetter) =>
      dispatchChartUIState({
        type: 'UPDATE',
        stateSetter,
      }),
    [],
  );

  return { chartUIState, updateUIState, loading };
};

import React, { useCallback } from 'react';
import type { ExperimentRunsChartsUIConfiguration } from '../../experiment-page/models/ExperimentPageUIState';
import { RunsChartsCardConfig } from '../runs-charts.types';
import { getUUID } from '../../../../common/utils/ActionUtils';

export type RunsChartsUIConfigurationSetter = (
  state: ExperimentRunsChartsUIConfiguration,
) => ExperimentRunsChartsUIConfiguration;

const RunsChartsUIConfigurationContext = React.createContext<
  (stateSetter: RunsChartsUIConfigurationSetter, isSeeding?: boolean) => void
>(() => {});

/**
 * Creates a localized context to manage the UI state of the runs charts.
 * Accepts a function to update the state object, coming from simple state setter or a reducer.
 */
export const RunsChartsUIConfigurationContextProvider = ({
  children,
  updateChartsUIState,
}: {
  children: React.ReactNode;
  updateChartsUIState: (stateSetter: RunsChartsUIConfigurationSetter, isSeeding?: boolean) => void;
}) => (
  <RunsChartsUIConfigurationContext.Provider value={updateChartsUIState}>
    {children}
  </RunsChartsUIConfigurationContext.Provider>
);

/**
 * Returns a function to update the current overarching UI state of the runs charts.
 */
export const useUpdateRunsChartsUIConfiguration = () => React.useContext(RunsChartsUIConfigurationContext);

export const useReorderRunsChartsFn = () => {
  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();

  return useCallback(
    (sourceChartUuid: string, targetChartUuid: string) => {
      updateChartsUIState((current) => {
        const newChartsOrder = current.compareRunCharts?.slice();
        const newSectionsState = current.compareRunSections?.slice();
        if (!newChartsOrder || !newSectionsState) {
          return current;
        }

        const indexSource = newChartsOrder.findIndex((c) => c.uuid === sourceChartUuid);
        const indexTarget = newChartsOrder.findIndex((c) => c.uuid === targetChartUuid);

        // If one of the charts is not found, do nothing
        if (indexSource < 0 || indexTarget < 0) {
          return current;
        }

        const sourceChart = newChartsOrder[indexSource];
        const targetChart = newChartsOrder[indexTarget];

        const isSameMetricSection = targetChart.metricSectionId === sourceChart.metricSectionId;

        // Update the sections to indicate that the charts have been reordered
        const sourceSectionIdx = newSectionsState.findIndex((c) => c.uuid === sourceChart.metricSectionId);
        const targetSectionIdx = newSectionsState.findIndex((c) => c.uuid === targetChart.metricSectionId);
        newSectionsState.splice(sourceSectionIdx, 1, { ...newSectionsState[sourceSectionIdx], isReordered: true });
        newSectionsState.splice(targetSectionIdx, 1, { ...newSectionsState[targetSectionIdx], isReordered: true });

        // Set new chart metric group
        const newSourceChart = { ...sourceChart };
        newSourceChart.metricSectionId = targetChart.metricSectionId;

        // Remove the source graph from array
        newChartsOrder.splice(indexSource, 1);
        if (!isSameMetricSection) {
          // Insert the source graph into target
          newChartsOrder.splice(
            newChartsOrder.findIndex((c) => c.uuid === targetChartUuid),
            0,
            newSourceChart,
          );
        } else {
          // The indexTarget is not neccessarily the target now, but it will work as the insert index
          newChartsOrder.splice(indexTarget, 0, newSourceChart);
        }

        return {
          ...current,
          compareRunCharts: newChartsOrder,
          compareRunSections: newSectionsState,
        };
      });
    },
    [updateChartsUIState],
  );
};

export const useConfirmChartCardConfigurationFn = () => {
  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();
  return useCallback(
    (configuredCard: Partial<RunsChartsCardConfig>) => {
      const serializedCard = RunsChartsCardConfig.serialize({
        ...configuredCard,
        uuid: getUUID(),
      });

      // Creating new chart
      if (!configuredCard.uuid) {
        updateChartsUIState((current) => ({
          ...current,
          // This condition ensures that chart collection will remain undefined if not set previously
          compareRunCharts: current.compareRunCharts && [...current.compareRunCharts, serializedCard],
        }));
      } /* Editing existing chart */ else {
        updateChartsUIState((current) => ({
          ...current,
          compareRunCharts: current.compareRunCharts?.map((existingChartCard) => {
            if (existingChartCard.uuid === configuredCard.uuid) {
              return { ...serializedCard, uuid: existingChartCard.uuid };
            }
            return existingChartCard;
          }),
        }));
      }
    },
    [updateChartsUIState],
  );
};

export const useInsertRunsChartsFn = () => {
  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();
  return useCallback(
    (sourceChartUuid: string, targetSectionId: string) => {
      updateChartsUIState((current) => {
        const newChartsOrder = current.compareRunCharts?.slice();
        const newSectionsState = current.compareRunSections?.slice();
        if (!newChartsOrder || !newSectionsState) {
          return current;
        }

        const indexSource = newChartsOrder.findIndex((c) => c.uuid === sourceChartUuid);
        if (indexSource < 0) {
          return current;
        }
        const sourceChart = newChartsOrder[indexSource];
        // Set new chart metric group
        const newSourceChart = { ...sourceChart };
        newSourceChart.metricSectionId = targetSectionId;

        // Update the sections to indicate that the charts have been reordered
        const sourceSectionIdx = newSectionsState.findIndex((c) => c.uuid === sourceChart.metricSectionId);
        const targetSectionIdx = newSectionsState.findIndex((c) => c.uuid === targetSectionId);
        newSectionsState.splice(sourceSectionIdx, 1, { ...newSectionsState[sourceSectionIdx], isReordered: true });
        newSectionsState.splice(targetSectionIdx, 1, { ...newSectionsState[targetSectionIdx], isReordered: true });

        // Remove the source graph from array and append
        newChartsOrder.splice(indexSource, 1);
        newChartsOrder.push(newSourceChart);

        return {
          ...current,
          compareRunCharts: newChartsOrder,
          compareRunSections: newSectionsState,
        };
      });
    },
    [updateChartsUIState],
  );
};

export const useRemoveRunsChartFn = () => {
  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();

  return useCallback(
    (configToDelete: RunsChartsCardConfig) => {
      updateChartsUIState((current) => ({
        ...current,
        compareRunCharts: configToDelete.isGenerated
          ? current.compareRunCharts?.map((setup) =>
              setup.uuid === configToDelete.uuid ? { ...setup, deleted: true } : setup,
            )
          : current.compareRunCharts?.filter((setup) => setup.uuid !== configToDelete.uuid),
      }));
    },
    [updateChartsUIState],
  );
};

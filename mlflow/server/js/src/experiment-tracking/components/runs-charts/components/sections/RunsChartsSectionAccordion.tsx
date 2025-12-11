import { Accordion } from '@databricks/design-system';
import type { ChartSectionConfig } from '../../../../types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsScatterCardConfig,
  RunsChartsContourCardConfig,
  SerializedRunsChartsCardConfigCard,
  RunsChartsParallelCardConfig,
} from '../../runs-charts.types';
import { RunsChartType } from '../../runs-charts.types';
import MetricChartsAccordion, { METRIC_CHART_SECTION_HEADER_SIZE } from '../../../MetricChartsAccordion';
import { RunsChartsSectionHeader } from './RunsChartsSectionHeader';
import { RunsChartsSection } from './RunsChartsSection';
import { useCallback, useMemo } from 'react';
import { getUUID } from '@mlflow/mlflow/src/common/utils/ActionUtils';
import { useState } from 'react';
import { Button, PlusIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Empty } from '@databricks/design-system';
import { useDesignSystemTheme } from '@databricks/design-system';
import { Spacer } from '@databricks/design-system';
import { useUpdateRunsChartsUIConfiguration } from '../../hooks/useRunsChartsUIConfiguration';
import { compact, isArray } from 'lodash';
import type { RunsChartCardSetFullscreenFn } from '../cards/ChartCard.common';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsGlobalLineChartConfig } from '../../../experiment-page/models/ExperimentPageUIState';

const chartMatchesFilter = (filter: string, config: RunsChartsCardConfig) => {
  // Use regexp-based filtering if a feature flag is enabled
  if (config.type === RunsChartType.IMAGE || config.type === RunsChartType.DIFFERENCE) {
    return true;
  }

  try {
    const filterRegex = new RegExp(filter, 'i');
    return getChartMetricsAndParams(config).some((metricOrParam) => metricOrParam.match(filterRegex));
  } catch {
    // If the regex is invalid (e.g. user it still typing it), prevent from filtering
    return true;
  }
};

const getChartMetricsAndParams = (config: RunsChartsCardConfig): string[] => {
  if (config.type === RunsChartType.BAR) {
    const barConfig = config as RunsChartsBarCardConfig;
    if (barConfig.dataAccessKey) {
      return [barConfig.metricKey, barConfig.dataAccessKey];
    }
    return [barConfig.metricKey];
  } else if (config.type === RunsChartType.LINE) {
    const lineConfig = config as RunsChartsLineCardConfig;
    if (isArray(lineConfig.selectedMetricKeys)) {
      return lineConfig.selectedMetricKeys;
    }
    return [lineConfig.metricKey];
  } else if (config.type === RunsChartType.SCATTER) {
    const scatterConfig = config as RunsChartsScatterCardConfig;
    return [scatterConfig.xaxis.key.toLowerCase(), scatterConfig.yaxis.key.toLowerCase()];
  } else if (config.type === RunsChartType.PARALLEL) {
    const parallelConfig = config as RunsChartsParallelCardConfig;
    return [...parallelConfig.selectedMetrics, ...parallelConfig.selectedParams];
  } else {
    const contourConfig = config as RunsChartsContourCardConfig;
    return [contourConfig.xaxis.key, contourConfig.yaxis.key, contourConfig.zaxis.key];
  }
};

export interface RunsChartsSectionAccordionProps {
  compareRunSections?: ChartSectionConfig[];
  compareRunCharts?: SerializedRunsChartsCardConfigCard[];
  reorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  insertCharts: (sourceChartUuid: string, targetSectionId: string) => void;
  chartData: RunsChartsRunData[];
  isMetricHistoryLoading?: boolean;
  startEditChart: (chartCard: RunsChartsCardConfig) => void;
  removeChart: (configToDelete: RunsChartsCardConfig) => void;
  addNewChartCard: (metricSectionId: string) => (type: RunsChartType) => void;
  search: string;
  groupBy: RunsGroupByConfig | null;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  supportedChartTypes?: RunsChartType[] | undefined;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
  noRunsSelectedEmptyState?: React.ReactElement;
}

export const RunsChartsSectionAccordion = ({
  compareRunSections,
  compareRunCharts,
  reorderCharts,
  insertCharts,
  chartData,
  isMetricHistoryLoading = false,
  autoRefreshEnabled = false,
  startEditChart,
  removeChart,
  addNewChartCard,
  search,
  groupBy,
  supportedChartTypes,
  hideEmptyCharts,
  setFullScreenChart = () => {},
  globalLineChartConfig,
  noRunsSelectedEmptyState,
}: RunsChartsSectionAccordionProps) => {
  const updateUIState = useUpdateRunsChartsUIConfiguration();
  const [editSection, setEditSection] = useState(-1);
  const { theme } = useDesignSystemTheme();

  /**
   * Get the active (expanded) panels for the accordion
   */
  const activeKey = useMemo(() => {
    const activeSections = (compareRunSections || []).flatMap((sectionConfig: ChartSectionConfig) => {
      if (sectionConfig.display) {
        return [sectionConfig.uuid];
      } else {
        return [];
      }
    });
    return activeSections;
  }, [compareRunSections]);

  /**
   * Updates the active (expanded) panels for the accordion
   */
  const onActivePanelChange = useCallback(
    (key: string | string[]) => {
      updateUIState((current) => {
        const newCompareRunPanels = (current.compareRunSections || []).map((sectionConfig: ChartSectionConfig) => {
          const sectionId = sectionConfig.uuid;
          const shouldDisplaySection =
            (typeof key === 'string' && sectionId === key) || (Array.isArray(key) && key.includes(sectionId));
          return {
            ...sectionConfig,
            display: shouldDisplaySection,
          };
        });
        return {
          ...current,
          compareRunSections: newCompareRunPanels,
        };
      });
    },
    [updateUIState],
  );

  /**
   * Deletes a section from the accordion
   */
  const deleteSection = useCallback(
    (sectionId: string) => {
      updateUIState((current) => {
        const newCompareRunCharts = (current.compareRunCharts || [])
          // Keep charts that are generated or not in section
          .filter((chartConfig: RunsChartsCardConfig) => {
            return chartConfig.isGenerated || chartConfig.metricSectionId !== sectionId;
          })
          // For charts that are generated and in section, set deleted to true
          .map((chartConfig: RunsChartsCardConfig) => {
            if (chartConfig.isGenerated && chartConfig.metricSectionId === sectionId) {
              return { ...chartConfig, deleted: true };
            } else {
              return chartConfig;
            }
          });

        // Delete section
        const newCompareRunSections = (current.compareRunSections || [])
          .slice()
          .filter((sectionConfig: ChartSectionConfig) => {
            return sectionConfig.uuid !== sectionId;
          });

        return {
          ...current,
          compareRunCharts: newCompareRunCharts,
          compareRunSections: newCompareRunSections,
          isAccordionReordered: true,
        };
      });
    },
    [updateUIState],
  );

  /**
   * Adds a section to the accordion
   * @param sectionId indicates the section selected to anchor at
   * @param above is a boolean value indicating whether to add the section above or below the anchor
   */
  const addSection = useCallback(
    (sectionId: string, above: boolean) => {
      let idx = -1;
      updateUIState((current) => {
        // Look for index
        const newCompareRunSections = [...(current.compareRunSections || [])];
        idx = newCompareRunSections.findIndex((sectionConfig: ChartSectionConfig) => sectionConfig.uuid === sectionId);
        const newSection = { name: '', uuid: getUUID(), display: false, isReordered: false };
        if (idx < 0) {
          // Index not found, add to end
          newCompareRunSections.push(newSection);
        } else if (above) {
          newCompareRunSections.splice(idx, 0, newSection);
        } else {
          idx += 1;
          newCompareRunSections.splice(idx, 0, newSection);
        }
        return {
          ...current,
          compareRunSections: newCompareRunSections,
          isAccordionReordered: true,
        };
      });
      return idx;
    },
    [updateUIState],
  );

  /**
   * Appends a section to the end of the accordion
   */
  const appendSection = useCallback(() => {
    updateUIState((current) => {
      const newCompareRunSections = [
        ...(current.compareRunSections || []),
        { name: '', uuid: getUUID(), display: false, isReordered: false },
      ];
      return {
        ...current,
        compareRunSections: newCompareRunSections,
        isAccordionReordered: true,
      };
    });
    setEditSection(compareRunSections?.length || -1);
  }, [updateUIState, compareRunSections?.length]);

  /**
   * Updates the name of a section
   * @param sectionId the section to update the name of
   * @param name the new name of the section
   */
  const setSectionName = useCallback(
    (sectionId: string, name: string) => {
      updateUIState((current) => {
        const newCompareRunSections = (current.compareRunSections || []).map((sectionConfig: ChartSectionConfig) => {
          if (sectionConfig.uuid === sectionId) {
            return { ...sectionConfig, name: name };
          } else {
            return sectionConfig;
          }
        });
        return {
          ...current,
          compareRunSections: newCompareRunSections,
          isAccordionReordered: true,
        };
      });
    },
    [updateUIState],
  );

  /**
   * Reorders the sections in the accordion
   * @param sourceSectionId the section you are dragging
   * @param targetSectionId the section to drop
   */
  const sectionReorder = useCallback(
    (sourceSectionId: string, targetSectionId: string) => {
      updateUIState((current) => {
        const newCompareRunSections = (current.compareRunSections || []).slice();
        const sourceSectionIdx = newCompareRunSections.findIndex(
          (sectionConfig: ChartSectionConfig) => sectionConfig.uuid === sourceSectionId,
        );
        const targetSectionIdx = newCompareRunSections.findIndex(
          (sectionConfig: ChartSectionConfig) => sectionConfig.uuid === targetSectionId,
        );
        const sourceSection = newCompareRunSections.splice(sourceSectionIdx, 1)[0];
        // If the source section is above the target section, the target section index will be shifted down by 1
        newCompareRunSections.splice(targetSectionIdx, 0, sourceSection);
        return {
          ...current,
          compareRunSections: newCompareRunSections,
          isAccordionReordered: true,
        };
      });
    },
    [updateUIState],
  );

  const noRunsSelected = useMemo(() => chartData.filter(({ hidden }) => !hidden).length === 0, [chartData]);

  const { sectionsToRender, chartsToRender } = useMemo(() => {
    if (search === '') {
      return { sectionsToRender: compareRunSections, chartsToRender: compareRunCharts };
    }

    const compareRunChartsFiltered = (compareRunCharts || []).filter((config: RunsChartsCardConfig) => {
      return !config.deleted && chartMatchesFilter(search, config);
    });
    // Get the sections that have these charts
    const sectionsWithCharts = new Set<string>();
    compareRunChartsFiltered.forEach((config: RunsChartsCardConfig) => {
      if (config.metricSectionId) {
        sectionsWithCharts.add(config.metricSectionId);
      }
    });
    // Filter the sections
    const compareRunSectionsFiltered = (compareRunSections || []).filter((sectionConfig: ChartSectionConfig) => {
      return sectionsWithCharts.has(sectionConfig.uuid);
    });

    return { sectionsToRender: compareRunSectionsFiltered, chartsToRender: compareRunChartsFiltered };
  }, [search, compareRunCharts, compareRunSections]);

  const isSearching = search !== '';

  if (!compareRunSections || !compareRunCharts) {
    return null;
  }

  if (noRunsSelected) {
    return (
      noRunsSelectedEmptyState ?? (
        <div css={{ marginTop: theme.spacing.lg }}>
          <Empty
            description={
              <FormattedMessage
                defaultMessage="All runs are hidden. Select at least one run to view charts."
                description="Experiment tracking > runs charts > indication displayed when no runs are selected for comparison"
              />
            }
          />
        </div>
      )
    );
  }

  if (isSearching && chartsToRender?.length === 0) {
    // Render empty in the center of the page
    return (
      <>
        <Spacer size="lg" />
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No metric charts"
              description="Experiment page > compare runs > no metric charts"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="All charts are filtered. Clear the search filter to see hidden metric charts."
              description="Experiment page > compare runs > no metric charts > description"
            />
          }
        />
      </>
    );
  }

  return (
    <div>
      <MetricChartsAccordion activeKey={activeKey} onActiveKeyChange={onActivePanelChange}>
        {(sectionsToRender || []).map((sectionConfig: ChartSectionConfig, index: number) => {
          const sectionCharts = (chartsToRender || []).filter((config: RunsChartsCardConfig) => {
            const section = (config as RunsChartsBarCardConfig).metricSectionId;
            return !config.deleted && section === sectionConfig.uuid;
          });

          return (
            <Accordion.Panel
              header={
                <RunsChartsSectionHeader
                  index={index}
                  section={sectionConfig}
                  onDeleteSection={deleteSection}
                  onAddSection={addSection}
                  editSection={editSection}
                  onSetEditSection={setEditSection}
                  onSetSectionName={setSectionName}
                  sectionChartsLength={sectionCharts.length}
                  addNewChartCard={addNewChartCard}
                  onSectionReorder={sectionReorder}
                  isExpanded={activeKey.includes(sectionConfig.uuid)}
                  supportedChartTypes={supportedChartTypes}
                  // When searching, hide the section placement controls
                  hideExtraControls={isSearching}
                />
              }
              key={sectionConfig.uuid}
              aria-hidden={!activeKey.includes(sectionConfig.uuid)}
            >
              <RunsChartsSection
                sectionId={sectionConfig.uuid}
                sectionConfig={sectionConfig}
                sectionCharts={sectionCharts}
                reorderCharts={reorderCharts}
                insertCharts={insertCharts}
                isMetricHistoryLoading={isMetricHistoryLoading}
                chartData={chartData}
                startEditChart={startEditChart}
                removeChart={removeChart}
                groupBy={groupBy}
                sectionIndex={index}
                setFullScreenChart={setFullScreenChart}
                autoRefreshEnabled={autoRefreshEnabled}
                hideEmptyCharts={hideEmptyCharts}
                globalLineChartConfig={globalLineChartConfig}
              />
            </Accordion.Panel>
          );
        })}
      </MetricChartsAccordion>
      {!isSearching && (
        <div>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_sections_runscomparesectionaccordion.tsx_405"
            block
            onClick={appendSection}
            icon={<PlusIcon />}
            style={{ border: 'none', marginTop: '6px' }}
          >
            <FormattedMessage
              defaultMessage="Add section"
              description="Experiment page > compare runs > chart section > add section bar"
            />
          </Button>
        </div>
      )}
    </div>
  );
};

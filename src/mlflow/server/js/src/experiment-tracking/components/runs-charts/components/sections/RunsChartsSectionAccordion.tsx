import { Accordion } from '@databricks/design-system';
import { ChartSectionConfig } from '../../../../types';
import { RunsChartsRunData } from '../RunsCharts.common';
import {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartType,
  RunsChartsLineCardConfig,
  RunsChartsScatterCardConfig,
  RunsChartsContourCardConfig,
  SerializedRunsChartsCardConfigCard,
} from '../../runs-charts.types';
import MetricChartsAccordion, { METRIC_CHART_SECTION_HEADER_SIZE } from '../../../MetricChartsAccordion';
import { RunsChartsSectionHeader } from './RunsChartsSectionHeader';
import { RunsChartsSection } from './RunsChartsSection';
import { useMemo } from 'react';
import { getUUID } from 'common/utils/ActionUtils';
import { useState } from 'react';
import { Button, PlusIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Empty } from '@databricks/design-system';
import { useDesignSystemTheme } from '@databricks/design-system';
import { Spacer } from '@databricks/design-system';
import { useUpdateRunsChartsUIConfiguration } from '../../hooks/useRunsChartsUIConfiguration';
import { isArray } from 'lodash';
import { RunsChartCardSetFullscreenFn } from '../cards/ChartCard.common';

const chartMatchesFilter = (filter: string, config: RunsChartsCardConfig) => {
  const filterLowerCase = filter.toLowerCase();

  if (config.type === RunsChartType.BAR) {
    const barConfig = config as RunsChartsBarCardConfig;
    return barConfig.metricKey.toLowerCase().includes(filterLowerCase);
  } else if (config.type === RunsChartType.LINE) {
    const lineConfig = config as RunsChartsLineCardConfig;
    if (isArray(lineConfig.selectedMetricKeys)) {
      return lineConfig.selectedMetricKeys.some((metricKey) => metricKey.toLowerCase().includes(filterLowerCase));
    }
    return lineConfig.metricKey.toLowerCase().includes(filterLowerCase);
  } else if (config.type === RunsChartType.SCATTER) {
    const scatterConfig = config as RunsChartsScatterCardConfig;
    return (
      scatterConfig.xaxis.key.toLowerCase().includes(filterLowerCase) ||
      scatterConfig.yaxis.key.toLowerCase().includes(filterLowerCase)
    );
  } else if (config.type === RunsChartType.PARALLEL) {
    return 'Parallel Coordinates'.toLowerCase().includes(filterLowerCase);
  } else {
    // Must be contour
    const contourConfig = config as RunsChartsContourCardConfig;
    return (
      contourConfig.xaxis.key.toLowerCase().includes(filterLowerCase) ||
      contourConfig.yaxis.key.toLowerCase().includes(filterLowerCase) ||
      contourConfig.zaxis.key.toLowerCase().includes(filterLowerCase)
    );
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
  groupBy?: string;
  supportedChartTypes?: RunsChartType[] | undefined;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
}

export const RunsChartsSectionAccordion = ({
  compareRunSections,
  compareRunCharts,
  reorderCharts,
  insertCharts,
  chartData,
  isMetricHistoryLoading = false,
  startEditChart,
  removeChart,
  addNewChartCard,
  search,
  groupBy = '',
  supportedChartTypes,
  setFullScreenChart = () => {},
}: RunsChartsSectionAccordionProps) => {
  const updateUIState = useUpdateRunsChartsUIConfiguration();
  const [editSection, setEditSection] = useState(-1);
  const { theme } = useDesignSystemTheme();

  // Filter the sections and chart by search

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
  const onActivePanelChange = (key: string | string[]) => {
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
  };

  /**
   * Deletes a section from the accordion
   */
  const deleteSection = (sectionId: string) => {
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
  };

  /**
   * Adds a section to the accordion
   * @param sectionId indicates the section selected to anchor at
   * @param above is a boolean value indicating whether to add the section above or below the anchor
   */
  const addSection = (sectionId: string, above: boolean) => {
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
  };

  /**
   * Appends a section to the end of the accordion
   */
  const appendSection = () => {
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
  };

  /**
   * Updates the name of a section
   * @param sectionId the section to update the name of
   * @param name the new name of the section
   */
  const setSectionName = (sectionId: string, name: string) => {
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
  };

  /**
   * Reorders the sections in the accordion
   * @param sourceSectionId the section you are dragging
   * @param targetSectionId the section to drop
   */
  const sectionReorder = (sourceSectionId: string, targetSectionId: string) => {
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
  };

  if (!compareRunSections || !compareRunCharts) {
    return null;
  }

  // If search is not empty, render the filtered charts
  if (search !== '') {
    const compareRunChartsFiltered = compareRunCharts.filter((config: RunsChartsCardConfig) => {
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
    const compareRunSectionsFiltered = compareRunSections.filter((sectionConfig: ChartSectionConfig) => {
      return sectionsWithCharts.has(sectionConfig.uuid);
    });

    if (compareRunChartsFiltered.length === 0) {
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
      <>
        <MetricChartsAccordion activeKey={compareRunSectionsFiltered.map(({ uuid }) => uuid)} disableCollapse>
          {compareRunSectionsFiltered.map((sectionConfig: ChartSectionConfig, index: number) => {
            const HEADING_PADDING_HEIGHT = 4;
            const EDITABLE_LABEL_PADDING_WIDTH = 6;
            const EDITABLE_LABEL_BORDER_WIDTH = 1;
            const EDITABLE_LABEL_OFFSET = EDITABLE_LABEL_PADDING_WIDTH + EDITABLE_LABEL_BORDER_WIDTH;
            // Get the charts in the section that are not deleted
            const filteredSectionCharts = compareRunChartsFiltered.filter((config: RunsChartsCardConfig) => {
              const section = (config as RunsChartsBarCardConfig).metricSectionId;
              return section === sectionConfig.uuid;
            });

            const runsCompareSearchHeader = (
              <div
                role="figure"
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  width: '100%',
                  padding: `${HEADING_PADDING_HEIGHT}px 0px`,
                  height: `${METRIC_CHART_SECTION_HEADER_SIZE}px`,
                }}
              >
                <div
                  css={{
                    paddingLeft: EDITABLE_LABEL_OFFSET,
                    whiteSpace: 'pre-wrap',
                  }}
                  data-testid="on-search-runs-compare-section-header"
                >
                  {sectionConfig.name}
                </div>
                <div
                  css={{
                    padding: theme.spacing.xs,
                    position: 'relative',
                  }}
                >
                  {`(${filteredSectionCharts.length})`}
                </div>
              </div>
            );

            return (
              <Accordion.Panel header={runsCompareSearchHeader} key={sectionConfig.uuid} collapsible="disabled">
                <RunsChartsSection
                  sectionId={sectionConfig.uuid}
                  sectionCharts={filteredSectionCharts}
                  reorderCharts={reorderCharts}
                  insertCharts={insertCharts}
                  isMetricHistoryLoading={isMetricHistoryLoading}
                  chartData={chartData}
                  startEditChart={startEditChart}
                  removeChart={removeChart}
                  groupBy={groupBy}
                  sectionIndex={index}
                  setFullScreenChart={setFullScreenChart}
                />
              </Accordion.Panel>
            );
          })}
        </MetricChartsAccordion>
      </>
    );
  }
  return (
    <div>
      <MetricChartsAccordion activeKey={activeKey} onActiveKeyChange={onActivePanelChange}>
        {(compareRunSections || []).map((sectionConfig: ChartSectionConfig, index: number) => {
          const sectionCharts = (compareRunCharts || []).filter((config: RunsChartsCardConfig) => {
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
                />
              }
              key={sectionConfig.uuid}
            >
              <RunsChartsSection
                sectionId={sectionConfig.uuid}
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
              />
            </Accordion.Panel>
          );
        })}
      </MetricChartsAccordion>
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
    </div>
  );
};

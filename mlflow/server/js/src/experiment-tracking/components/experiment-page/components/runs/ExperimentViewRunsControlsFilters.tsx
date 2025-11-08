import {
  Button,
  Tag,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSelectItem,
  DialogComboboxOptionListSearch,
  DialogComboboxTrigger,
  DownloadIcon,
  ClipboardIcon,
  FullscreenExitIcon,
  FullscreenIcon,
  OverflowIcon,
  PlusIcon,
  SidebarIcon,
  LegacyTooltip,
  useDesignSystemTheme,
  DropdownMenu,
  ToggleButton,
  SegmentedControlGroup,
  SegmentedControlButton,
  ListIcon,
  Tooltip,
  ChartLineIcon,
  TableIcon,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';

import { shouldEnablePromptLab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import React, { useMemo, useState } from 'react';
import { useSelector } from 'react-redux';
import { FormattedMessage, useIntl } from 'react-intl';
import { ToggleIconButton } from '../../../../../common/components/ToggleIconButton';
import type { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { LIFECYCLE_FILTER } from '../../../../constants';
import type { UpdateExperimentViewStateFn } from '../../../../types';
import { useExperimentIds } from '../../hooks/useExperimentIds';
import type { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { getStartTimeColumnDisplayName } from '../../utils/experimentPage.common-utils';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';
import type { ExperimentStoreEntities, DatasetSummary, ExperimentViewRunsCompareMode } from '../../../../types';
import { datasetSummariesEqual } from '../../../../utils/DatasetUtils';
import { CreateNotebookRunModal } from '@mlflow/mlflow/src/experiment-tracking/components/evaluation-artifacts-compare/CreateNotebookRunModal';
import { PreviewBadge } from '@mlflow/mlflow/src/shared/building_blocks/PreviewBadge';
import { useCreateNewRun } from '../../hooks/useCreateNewRun';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import { useUpdateExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';

export type ExperimentViewRunsControlsFiltersProps = {
  searchFacetsState: ExperimentPageSearchFacetsState;
  experimentId: string;
  viewState: ExperimentPageViewState;
  updateViewState: UpdateExperimentViewStateFn;
  runsData: ExperimentRunsSelectorResult;
  onDownloadCsv: () => void;
  requestError: ErrorWrapper | Error | null;
  additionalControls?: React.ReactNode;
  refreshRuns: () => void;
  viewMaximized: boolean;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  areRunsGrouped?: boolean;
};

export const ExperimentViewRunsControlsFilters = React.memo(
  ({
    searchFacetsState,
    experimentId,
    runsData,
    viewState,
    updateViewState,
    onDownloadCsv,
    requestError,
    additionalControls,
    refreshRuns,
    viewMaximized,
    autoRefreshEnabled = false,
    hideEmptyCharts = false,
    areRunsGrouped = false,
  }: ExperimentViewRunsControlsFiltersProps) => {
    const setUrlSearchFacets = useUpdateExperimentPageSearchFacets();

    const [pageViewMode, setViewModeInURL] = useExperimentPageViewMode();
    const updateUIState = useUpdateExperimentViewUIState();

    const isComparingExperiments = useExperimentIds().length > 1;
    const { startTime, lifecycleFilter, datasetsFilter, searchFilter } = searchFacetsState;

    // Use modernized view mode value getter if flag is set
    const compareRunsMode = pageViewMode;

    const intl = useIntl();
    const { createNewRun } = useCreateNewRun();
    const [isCreateRunWithNotebookModalOpen, setCreateRunWithNotebookModalOpenValue] = useState(false);
    const { theme } = useDesignSystemTheme();

    // List of labels for "start time" filter
    const startTimeColumnLabels: Record<string, string> = useMemo(() => getStartTimeColumnDisplayName(intl), [intl]);

    const currentLifecycleFilterValue =
      lifecycleFilter === LIFECYCLE_FILTER.ACTIVE
        ? intl.formatMessage({
            defaultMessage: 'Active',
            description: 'Linked model dropdown option to show active experiment runs',
          })
        : intl.formatMessage({
            defaultMessage: 'Deleted',
            description: 'Linked model dropdown option to show deleted experiment runs',
          });

    const currentStartTimeFilterLabel = intl.formatMessage({
      defaultMessage: 'Time created',
      description: 'Label for the start time select dropdown for experiment runs view',
    });

    // Show preview sidebar only on table view and artifact view
    const displaySidebarToggleButton = compareRunsMode === undefined || compareRunsMode === 'ARTIFACT';

    const datasetSummaries: DatasetSummary[] = useSelector(
      (state: { entities: ExperimentStoreEntities }) => state.entities.datasetsByExperimentId[experimentId],
    );

    const updateDatasetsFilter = (summary: DatasetSummary) => {
      const newDatasetsFilter = datasetsFilter.some((item) => datasetSummariesEqual(item, summary))
        ? datasetsFilter.filter((item) => !datasetSummariesEqual(item, summary))
        : [...datasetsFilter, summary];

      setUrlSearchFacets({
        datasetsFilter: newDatasetsFilter,
      });
    };

    const hasDatasets = datasetSummaries !== undefined;

    const searchFilterChange = (newSearchFilter: string) => {
      setUrlSearchFacets({ searchFilter: newSearchFilter });
    };

    return (
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          justifyContent: 'space-between',
          [theme.responsive.mediaQueries.xs]: {
            flexDirection: 'column',
          },
        }}
      >
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.sm,
            alignItems: 'center',
            flexWrap: 'wrap' as const,
          }}
        >
          <SegmentedControlGroup
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_184"
            name="runs-view-mode"
            value={pageViewMode}
            onChange={({ target }) => {
              const { value } = target;
              const newValue = value as ExperimentViewRunsCompareMode;

              if (pageViewMode === newValue) {
                return;
              }

              setViewModeInURL(newValue);
            }}
          >
            <SegmentedControlButton
              value="TABLE"
              icon={
                <Tooltip
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_201"
                  content={intl.formatMessage({
                    defaultMessage: 'Table view',
                    description: 'Experiment page > control bar > table view toggle button tooltip',
                  })}
                >
                  <ListIcon />
                </Tooltip>
              }
            />
            <SegmentedControlButton
              value="CHART"
              icon={
                <Tooltip
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_211"
                  content={intl.formatMessage({
                    defaultMessage: 'Chart view',
                    description: 'Experiment page > control bar > chart view toggle button tooltip',
                  })}
                >
                  <ChartLineIcon />
                </Tooltip>
              }
            />
            <SegmentedControlButton
              value="ARTIFACT"
              disabled={areRunsGrouped}
              icon={
                <Tooltip
                  componentId="mlflow.experiment_page.mode.artifact"
                  content={
                    areRunsGrouped
                      ? intl.formatMessage({
                          defaultMessage: 'Unavailable when runs are grouped',
                          description: 'Experiment page > view mode switch > evaluation mode disabled tooltip',
                        })
                      : intl.formatMessage({
                          defaultMessage: 'Artifact evaluation',
                          description:
                            'A tooltip for the view mode switcher in the experiment view, corresponding to artifact evaluation view',
                        })
                  }
                >
                  <TableIcon />
                </Tooltip>
              }
            />
          </SegmentedControlGroup>

          <RunsSearchAutoComplete
            runsData={runsData}
            searchFilter={searchFilter}
            onSearchFilterChange={searchFilterChange}
            onClear={() => {
              setUrlSearchFacets(createExperimentPageSearchFacetsState());
            }}
            requestError={requestError}
          />

          <DialogCombobox
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_217"
            label={currentStartTimeFilterLabel}
            value={startTime !== 'ALL' ? [startTimeColumnLabels[startTime]] : []}
          >
            <DialogComboboxTrigger
              allowClear={startTime !== 'ALL'}
              onClear={() => {
                setUrlSearchFacets({ startTime: 'ALL' });
              }}
              data-testid="start-time-select-dropdown"
            />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {Object.keys(startTimeColumnLabels).map((startTimeKey) => (
                  <DialogComboboxOptionListSelectItem
                    key={startTimeKey}
                    checked={startTimeKey === startTime}
                    title={startTimeColumnLabels[startTimeKey]}
                    data-testid={`start-time-select-${startTimeKey}`}
                    value={startTimeKey}
                    onChange={() => {
                      setUrlSearchFacets({ startTime: startTimeKey });
                    }}
                  >
                    {startTimeColumnLabels[startTimeKey]}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>

          <DialogCombobox
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_248"
            label={intl.formatMessage({
              defaultMessage: 'State',
              description: 'Filtering label to filter experiments based on state of active or deleted',
            })}
            value={[currentLifecycleFilterValue]}
          >
            <DialogComboboxTrigger allowClear={false} data-testid="lifecycle-filter" />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSelectItem
                  checked={lifecycleFilter === LIFECYCLE_FILTER.ACTIVE}
                  key={LIFECYCLE_FILTER.ACTIVE}
                  data-testid="active-runs-menu-item"
                  value={LIFECYCLE_FILTER.ACTIVE}
                  onChange={() => {
                    setUrlSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.ACTIVE });
                  }}
                >
                  <FormattedMessage
                    defaultMessage="Active"
                    description="Linked model dropdown option to show active experiment runs"
                  />
                </DialogComboboxOptionListSelectItem>
                <DialogComboboxOptionListSelectItem
                  checked={lifecycleFilter === LIFECYCLE_FILTER.DELETED}
                  key={LIFECYCLE_FILTER.DELETED}
                  data-testid="deleted-runs-menu-item"
                  value={LIFECYCLE_FILTER.DELETED}
                  onChange={() => {
                    setUrlSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.DELETED });
                  }}
                >
                  <FormattedMessage
                    defaultMessage="Deleted"
                    description="Linked model dropdown option to show deleted experiment runs"
                  />
                </DialogComboboxOptionListSelectItem>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
          <DialogCombobox
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_289"
            label={intl.formatMessage({
              defaultMessage: 'Datasets',
              description: 'Filtering label to filter runs based on datasets used',
            })}
            value={datasetsFilter.map((datasetSummary) => datasetSummary.name)}
            multiSelect
          >
            <LegacyTooltip
              title={
                !hasDatasets && (
                  <FormattedMessage
                    defaultMessage="No datasets were recorded for this experiment's runs."
                    description="Message to indicate that no datasets were recorded for this experiment's runs."
                  />
                )
              }
            >
              <DialogComboboxTrigger
                allowClear
                onClear={() => setUrlSearchFacets({ datasetsFilter: [] })}
                data-testid="datasets-select-dropdown"
                showTagAfterValueCount={1}
                disabled={!hasDatasets}
              />
              {hasDatasets && (
                <DialogComboboxContent maxHeight={600}>
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListSearch>
                      {datasetSummaries.map((summary: DatasetSummary) => (
                        <DialogComboboxOptionListCheckboxItem
                          key={summary.name + summary.digest + summary.context}
                          checked={datasetsFilter.some((item) => datasetSummariesEqual(item, summary))}
                          title={summary.name}
                          data-testid={`dataset-dropdown-${summary.name}`}
                          value={summary.name}
                          onChange={() => updateDatasetsFilter(summary)}
                        >
                          {summary.name} ({summary.digest}){' '}
                          {summary.context && (
                            <Tag
                              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_329"
                              css={{ textTransform: 'capitalize', marginRight: theme.spacing.xs }}
                            >
                              {summary.context}
                            </Tag>
                          )}
                        </DialogComboboxOptionListCheckboxItem>
                      ))}
                    </DialogComboboxOptionListSearch>
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              )}
            </LegacyTooltip>
          </DialogCombobox>
          {additionalControls}
        </div>
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.sm,
            alignItems: 'flex-start',
          }}
        >
          <DropdownMenu.Root modal={false}>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_338"
                icon={<OverflowIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'More options',
                  description: 'Experiment page > control bar > more options button accessible label',
                })}
              />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_362"
                className="csv-button"
                onClick={onDownloadCsv}
              >
                <DropdownMenu.IconWrapper>
                  <DownloadIcon />
                </DropdownMenu.IconWrapper>
                {`Download ${runsData.runInfos.length} runs`}
              </DropdownMenu.Item>
              <DropdownMenu.Separator />
              <DropdownMenu.CheckboxItem
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_382"
                checked={hideEmptyCharts}
                onClick={() =>
                  updateUIState((state) => ({
                    ...state,
                    hideEmptyCharts: !state.hideEmptyCharts,
                  }))
                }
              >
                <DropdownMenu.ItemIndicator />
                <FormattedMessage
                  defaultMessage="Hide charts with no data"
                  description="Experiment page > control bar > label for a checkbox toggle button that hides chart cards with no corresponding data"
                />
              </DropdownMenu.CheckboxItem>

              <DropdownMenu.Separator />
              <DropdownMenu.CheckboxItem
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_402"
                checked={autoRefreshEnabled}
                onClick={() =>
                  updateUIState((state) => ({
                    ...state,
                    autoRefreshEnabled: !state.autoRefreshEnabled,
                  }))
                }
              >
                <DropdownMenu.ItemIndicator />
                <FormattedMessage
                  defaultMessage="Auto-refresh"
                  description="String for the auto-refresh button that refreshes the runs list automatically"
                />
              </DropdownMenu.CheckboxItem>
            </DropdownMenu.Content>
          </DropdownMenu.Root>

          <CreateNotebookRunModal
            isOpen={isCreateRunWithNotebookModalOpen}
            closeModal={() => setCreateRunWithNotebookModalOpenValue(false)}
            experimentId={experimentId}
          />

          {displaySidebarToggleButton && (
            <LegacyTooltip
              title={intl.formatMessage({
                defaultMessage: 'Toggle the preview sidepane',
                description: 'Experiment page > control bar > expanded view toggle button tooltip',
              })}
              useAsLabel
            >
              <ToggleIconButton
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_403"
                pressed={viewState.previewPaneVisible}
                icon={<SidebarIcon />}
                onClick={() => updateViewState({ previewPaneVisible: !viewState.previewPaneVisible })}
              />
            </LegacyTooltip>
          )}
          {/* TODO: Add tooltip to guide users to this button */}
          {!isComparingExperiments && (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_415"
                  icon={<PlusIcon />}
                >
                  <FormattedMessage
                    defaultMessage="New run"
                    description="Button used to pop up a modal to create a new run"
                  />
                </Button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Content>
                {shouldEnablePromptLab() && (
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_461"
                    onSelect={() => createNewRun()}
                  >
                    {' '}
                    <FormattedMessage
                      defaultMessage="using Prompt Engineering"
                      description="String for creating a new run with prompt engineering modal"
                    />
                    <PreviewBadge />
                  </DropdownMenu.Item>
                )}
                <DropdownMenu.Item
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_469"
                  onSelect={() => setCreateRunWithNotebookModalOpenValue(true)}
                >
                  {' '}
                  <FormattedMessage
                    defaultMessage="using Notebook"
                    description="String for creating a new run from a notebook"
                  />
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          )}
        </div>
      </div>
    );
  },
);

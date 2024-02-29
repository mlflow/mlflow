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
  Tooltip,
  useDesignSystemTheme,
  DropdownMenu,
  ToggleButton,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import {
  shouldEnableExperimentPageCompactHeader,
  shouldEnablePromptLab,
  shouldEnableShareExperimentViewByTags,
} from 'common/utils/FeatureUtils';
import React, { useMemo, useState } from 'react';
import { useSelector } from 'react-redux';
import { FormattedMessage, useIntl } from 'react-intl';
import { ToggleIconButton } from '../../../../../common/components/ToggleIconButton';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { LIFECYCLE_FILTER } from '../../../../constants';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { useExperimentIds } from '../../hooks/useExperimentIds';
import {
  SearchExperimentRunsFacetsState,
  clearSearchExperimentsFacetsFilters,
} from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { getStartTimeColumnDisplayName } from '../../utils/experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRefreshButton } from './ExperimentViewRefreshButton';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';
import type { ExperimentStoreEntities, DatasetSummary } from '../../../../types';
import { datasetSummariesEqual } from '../../../../utils/DatasetUtils';
import { CreateNotebookRunModal } from 'experiment-tracking/components/evaluation-artifacts-compare/CreateNotebookRunModal';
import { PreviewBadge } from 'shared/building_blocks/PreviewBadge';
import { useCreateNewRun } from '../../hooks/useCreateNewRun';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import { useUpdateExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';
import { createExperimentPageSearchFacetsStateV2 } from '../../models/ExperimentPageSearchFacetsStateV2';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';

export type ExperimentViewRunsControlsFiltersProps = {
  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  experimentId: string;
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;
  runsData: ExperimentRunsSelectorResult;
  onDownloadCsv: () => void;
  requestError: ErrorWrapper | null;
  additionalControls?: React.ReactNode;
  refreshRuns: () => void;
  viewMaximized: boolean;
};

export const ExperimentViewRunsControlsFilters = React.memo(
  ({
    searchFacetsState,
    updateSearchFacets,
    experimentId,
    runsData,
    viewState,
    updateViewState,
    onDownloadCsv,
    requestError,
    additionalControls,
    refreshRuns,
    viewMaximized,
  }: ExperimentViewRunsControlsFiltersProps) => {
    const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
    const setUrlSearchFacets = useUpdateExperimentPageSearchFacets();

    const [pageViewMode] = useExperimentPageViewMode();
    const updateUIState = useUpdateExperimentViewUIState();

    const isComparingExperiments = useExperimentIds().length > 1;
    const { startTime, lifecycleFilter, datasetsFilter, searchFilter } = searchFacetsState;

    // Use modernized view mode value getter if flag is set
    const compareRunsMode = usingNewViewStateModel ? pageViewMode : searchFacetsState.compareRunsMode;

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

      if (usingNewViewStateModel) {
        // In the new view state version, set datasets filter directly in the URL search state
        setUrlSearchFacets({
          datasetsFilter: newDatasetsFilter,
        });
      } else {
        updateSearchFacets((existingFacets) => ({
          ...existingFacets,
          datasetsFilter: newDatasetsFilter,
        }));
      }
    };

    const hasDatasets = datasetSummaries !== undefined;

    const searchFilterChange = (newSearchFilter: string) => {
      if (usingNewViewStateModel) {
        // In the new view state version, set search filter directly in the URL search state
        setUrlSearchFacets({ searchFilter: newSearchFilter });
      } else {
        updateSearchFacets({ searchFilter: newSearchFilter });
      }
    };

    return (
      <div css={styles.groupBar}>
        <div css={styles.controlBar}>
          <RunsSearchAutoComplete
            runsData={runsData}
            searchFilter={searchFilter}
            onSearchFilterChange={searchFilterChange}
            onClear={() => {
              if (usingNewViewStateModel) {
                // In the new view state version, reset URL search state directly
                setUrlSearchFacets(createExperimentPageSearchFacetsStateV2());
              } else {
                updateSearchFacets(clearSearchExperimentsFacetsFilters);
              }
            }}
            requestError={requestError}
          />

          <DialogCombobox
            label={currentStartTimeFilterLabel}
            value={startTime !== 'ALL' ? [startTimeColumnLabels[startTime]] : []}
          >
            <DialogComboboxTrigger
              allowClear={startTime !== 'ALL'}
              onClear={() => {
                if (usingNewViewStateModel) {
                  // In the new view state version, set time filter directly in the URL search state
                  setUrlSearchFacets({ startTime: 'ALL' });
                } else {
                  updateSearchFacets({ startTime: 'ALL' });
                }
              }}
              data-test-id="start-time-select-dropdown"
            />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {Object.keys(startTimeColumnLabels).map((startTimeKey) => (
                  <DialogComboboxOptionListSelectItem
                    key={startTimeKey}
                    checked={startTimeKey === startTime}
                    title={startTimeColumnLabels[startTimeKey]}
                    data-test-id={`start-time-select-${startTimeKey}`}
                    value={startTimeKey}
                    onChange={() => {
                      if (usingNewViewStateModel) {
                        // In the new view state version, set time filter directly in the URL search state
                        setUrlSearchFacets({ startTime: startTimeKey });
                      } else {
                        updateSearchFacets({ startTime: startTimeKey });
                      }
                    }}
                  >
                    {startTimeColumnLabels[startTimeKey]}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>

          <DialogCombobox
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
                    if (usingNewViewStateModel) {
                      // In the new view state version, set lifecycle filter directly in the URL search state
                      setUrlSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.ACTIVE });
                    } else {
                      updateSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.ACTIVE });
                    }
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
                    if (usingNewViewStateModel) {
                      // In the new view state version, set lifecycle filter directly in the URL search state
                      setUrlSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.DELETED });
                    } else {
                      updateSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.DELETED });
                    }
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
            label={intl.formatMessage({
              defaultMessage: 'Datasets',
              description: 'Filtering label to filter runs based on datasets used',
            })}
            value={datasetsFilter.map((datasetSummary) => datasetSummary.name)}
            multiSelect
          >
            <Tooltip
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
                onClear={() => updateSearchFacets({ datasetsFilter: [] })}
                data-test-id="datasets-select-dropdown"
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
                          data-test-id={`dataset-dropdown-${summary.name}`}
                          value={summary.name}
                          onChange={() => updateDatasetsFilter(summary)}
                        >
                          {summary.name} ({summary.digest}){' '}
                          {summary.context && (
                            <Tag css={{ textTransform: 'capitalize', marginRight: theme.spacing.xs }}>
                              {summary.context}
                            </Tag>
                          )}
                        </DialogComboboxOptionListCheckboxItem>
                      ))}
                    </DialogComboboxOptionListSearch>
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              )}
            </Tooltip>
          </DialogCombobox>
          {additionalControls}
        </div>
        <div css={styles.groupSeparator} />
        <div css={styles.controlBar}>
          <DropdownMenu.Root>
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
                className="csv-button"
                onClick={onDownloadCsv}
                css={{ display: 'flex', gap: theme.spacing.sm }}
              >
                <DownloadIcon />
                {`Download ${runsData.runInfos.length} runs`}
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>

          <CreateNotebookRunModal
            isOpen={isCreateRunWithNotebookModalOpen}
            closeModal={() => setCreateRunWithNotebookModalOpenValue(false)}
            experimentId={experimentId}
          />

          {!isComparingExperiments && !shouldEnableExperimentPageCompactHeader() && (
            /* 
          When comparing experiments, elements that are hidden upon 
          maximization are not displayed anyway so let's hide the button then
         */
            <Tooltip
              key={viewState.viewMaximized.toString()}
              title={intl.formatMessage(
                {
                  defaultMessage: 'Click to {isMaximized, select, true {restore} other {maximize}} the view',
                  description: 'Experiment page > control bar > expanded view toggle button tooltip',
                },
                {
                  isMaximized: viewState.viewMaximized,
                },
              )}
              useAsLabel
            >
              <ToggleIconButton
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_380"
                pressed={viewMaximized}
                icon={viewMaximized ? <FullscreenExitIcon /> : <FullscreenIcon />}
                onClick={() => {
                  if (usingNewViewStateModel) {
                    // In the new view state version, toggle view maximized in UI state object
                    updateUIState((state) => ({ ...state, viewMaximized: !state.viewMaximized }));
                  } else {
                    updateViewState({ viewMaximized: !viewState.viewMaximized });
                  }
                }}
              />
            </Tooltip>
          )}
          {displaySidebarToggleButton && (
            <Tooltip
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
            </Tooltip>
          )}
          <ExperimentViewRefreshButton refreshRuns={refreshRuns} />
          {/* TODO: Add tooltip to guide users to this button */}
          {shouldEnablePromptLab() && !isComparingExperiments && (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsfilters.tsx_415"
                  type="primary"
                  icon={<PlusIcon />}
                >
                  <FormattedMessage
                    defaultMessage="New run"
                    description="Button used to pop up a modal to create a new run"
                  />
                </Button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Content>
                <DropdownMenu.Item onSelect={() => createNewRun()}>
                  {' '}
                  <FormattedMessage
                    defaultMessage="using Prompt Engineering"
                    description="String for creating a new run with prompt engineering modal"
                  />
                  <PreviewBadge />
                </DropdownMenu.Item>
                <DropdownMenu.Item onSelect={() => setCreateRunWithNotebookModalOpenValue(true)}>
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

const styles = {
  groupBar: { display: 'grid', gridTemplateColumns: 'auto 1fr auto' },
  controlBar: (theme: Theme) => ({
    display: 'flex',
    gap: theme.spacing.sm,
    alignItems: 'center',
    flexWrap: 'wrap' as const,
  }),
  groupSeparator: (theme: Theme) => ({ minWidth: theme.spacing.sm }),
  columnSwitch: { margin: '5px' },
  searchBox: (theme: Theme) => ({ display: 'flex', gap: theme.spacing.sm, width: 560 }),
  lifecycleFilters: (theme: Theme) => ({
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    marginTop: theme.spacing.sm,
    marginBottom: theme.spacing.sm,
    marginLeft: theme.spacing.lg * 2,
  }),
};

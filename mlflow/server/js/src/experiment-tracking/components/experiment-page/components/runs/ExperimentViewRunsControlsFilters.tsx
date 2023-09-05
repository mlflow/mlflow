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
  Dropdown,
  FullscreenExitIcon,
  FullscreenIcon,
  Menu,
  OverflowIcon,
  PlusIcon,
  SidebarIcon,
  Tooltip,
  useDesignSystemTheme,
  DropdownMenu,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import {
  shouldEnableArtifactBasedEvaluation,
  shouldEnablePromptLab,
  shouldEnableDatasetsDropdown,
} from 'common/utils/FeatureUtils';
import React, { useCallback, useMemo, useState } from 'react';
import { useSelector } from 'react-redux';
import { FormattedMessage, useIntl } from 'react-intl';
import { ToggleIconButton } from '../../../../../common/components/ToggleIconButton';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { LIFECYCLE_FILTER } from '../../../../constants';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { useExperimentIds } from '../../hooks/useExperimentIds';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { getStartTimeColumnDisplayName } from '../../utils/experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRefreshButton } from './ExperimentViewRefreshButton';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';
import type { ExperimentStoreEntities, DatasetSummary } from '../../../../types';
import { datasetSummariesEqual } from '../../../../utils/DatasetUtils';
import { Data } from 'vega';
import { CreateNotebookRunModal } from 'experiment-tracking/components/evaluation-artifacts-compare/CreateNotebookRunModal';
import { useCreateNewRun } from '../../hooks/useCreateNewRun';

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
  }: ExperimentViewRunsControlsFiltersProps) => {
    const isComparingExperiments = useExperimentIds().length > 1;
    const { compareRunsMode, startTime, lifecycleFilter, datasetsFilter } = searchFacetsState;
    const intl = useIntl();
    const { createNewRun } = useCreateNewRun();
    const [isCreateRunWithNotebookModalOpen, setCreateRunWithNotebookModalOpenValue] =
      useState(false);
    const { theme } = useDesignSystemTheme();

    // List of labels for "start time" filter
    const startTimeColumnLabels: Record<string, string> = useMemo(
      () => getStartTimeColumnDisplayName(intl),
      [intl],
    );

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

    const currentStartTimeFilterLabel = (
      <>
        <FormattedMessage
          defaultMessage='Time created'
          description='Label for the start time select dropdown for experiment runs view'
        />
      </>
    );

    // Show preview sidebar only on table view and artifact view
    const displaySidebarToggleButton =
      compareRunsMode === undefined || compareRunsMode === 'ARTIFACT';

    const datasetSummaries: DatasetSummary[] = useSelector(
      (state: { entities: ExperimentStoreEntities }) =>
        state.entities.datasetsByExperimentId[experimentId],
    );

    const updateDatasetsFilter = useCallback(
      (summary: DatasetSummary) => {
        updateSearchFacets((existingFacets) => ({
          ...existingFacets,
          datasetsFilter: datasetsFilter.some((item) => datasetSummariesEqual(item, summary))
            ? datasetsFilter.filter((item) => !datasetSummariesEqual(item, summary))
            : [...datasetsFilter, summary],
        }));
      },
      [updateSearchFacets, datasetsFilter],
    );

    const hasDatasets = datasetSummaries !== undefined;

    return (
      <div css={styles.groupBar}>
        <div css={styles.controlBar}>
          <RunsSearchAutoComplete
            runsData={runsData}
            searchFacetsState={searchFacetsState}
            updateSearchFacets={updateSearchFacets}
            requestError={requestError}
          />

          <DialogCombobox
            label={currentStartTimeFilterLabel}
            value={startTime !== 'ALL' ? [startTimeColumnLabels[startTime]] : []}
          >
            <DialogComboboxTrigger
              allowClear={startTime !== 'ALL'}
              onClear={() => updateSearchFacets({ startTime: 'ALL' })}
              data-test-id='start-time-select-dropdown'
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
                    onChange={() => updateSearchFacets({ startTime: startTimeKey })}
                  >
                    {startTimeColumnLabels[startTimeKey]}
                  </DialogComboboxOptionListSelectItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>

          <DialogCombobox
            label={
              <FormattedMessage
                defaultMessage='State'
                description='Filtering label to filter experiments based on state of active or deleted'
              />
            }
            value={[currentLifecycleFilterValue]}
          >
            <DialogComboboxTrigger allowClear={false} data-testid='lifecycle-filter' />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSelectItem
                  checked={lifecycleFilter === LIFECYCLE_FILTER.ACTIVE}
                  key={LIFECYCLE_FILTER.ACTIVE}
                  data-testid='active-runs-menu-item'
                  value={LIFECYCLE_FILTER.ACTIVE}
                  onChange={() => updateSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.ACTIVE })}
                >
                  <FormattedMessage
                    defaultMessage='Active'
                    description='Linked model dropdown option to show active experiment runs'
                  />
                </DialogComboboxOptionListSelectItem>
                <DialogComboboxOptionListSelectItem
                  checked={lifecycleFilter === LIFECYCLE_FILTER.DELETED}
                  key={LIFECYCLE_FILTER.DELETED}
                  data-testid='deleted-runs-menu-item'
                  value={LIFECYCLE_FILTER.DELETED}
                  onChange={() => updateSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.DELETED })}
                >
                  <FormattedMessage
                    defaultMessage='Deleted'
                    description='Linked model dropdown option to show deleted experiment runs'
                  />
                </DialogComboboxOptionListSelectItem>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
          {shouldEnableDatasetsDropdown() && (
            <DialogCombobox
              label={
                <FormattedMessage
                  defaultMessage='Datasets'
                  description='Filtering label to filter runs based on datasets used'
                />
              }
              value={datasetsFilter.map((datasetSummary) => datasetSummary.name)}
              multiSelect
            >
              <Tooltip
                title={
                  !hasDatasets && (
                    <FormattedMessage
                      defaultMessage="No datasets registered for this experiment's runs."
                      description={'No datasets registered message for datasets dropdown.'}
                    />
                  )
                }
              >
                <DialogComboboxTrigger
                  allowClear
                  onClear={() => updateSearchFacets({ datasetsFilter: [] })}
                  data-test-id='datasets-select-dropdown'
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
                            checked={datasetsFilter.some((item) =>
                              datasetSummariesEqual(item, summary),
                            )}
                            title={summary.name}
                            data-test-id={`dataset-dropdown-${summary.name}`}
                            value={summary.name}
                            onChange={() => updateDatasetsFilter(summary)}
                          >
                            {summary.name} ({summary.digest}){' '}
                            {summary.context && (
                              <Tag
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
              </Tooltip>
            </DialogCombobox>
          )}
          {additionalControls}
        </div>
        <div css={styles.groupSeparator} />
        <div css={styles.controlBar}>
          <Dropdown
            trigger={['click']}
            placement='bottomRight'
            overlay={
              <Menu>
                <Menu.Item className='csv-button' onClick={onDownloadCsv}>
                  <DownloadIcon />{' '}
                  <FormattedMessage
                    defaultMessage='Download CSV'
                    description='String for the download csv button to download experiments offline in a CSV format'
                  />
                </Menu.Item>
              </Menu>
            }
          >
            <Button type='tertiary' icon={<OverflowIcon />} />
          </Dropdown>

          <CreateNotebookRunModal
            isOpen={isCreateRunWithNotebookModalOpen}
            closeModal={() => setCreateRunWithNotebookModalOpenValue(false)}
            experimentId={experimentId}
          />

          {!isComparingExperiments && (
            /* 
          When comparing experiments, elements that are hidden upon 
          maximization are not displayed anyway so let's hide the button then
         */
            <Tooltip
              key={viewState.viewMaximized.toString()}
              title={
                <FormattedMessage
                  defaultMessage='Click to {isMaximized, select, true {restore} other {maximize}} the view'
                  description='Experiment page > control bar > expanded view toggle button tooltip'
                  values={{
                    isMaximized: viewState.viewMaximized,
                  }}
                />
              }
            >
              <ToggleIconButton
                pressed={viewState.viewMaximized}
                icon={viewState.viewMaximized ? <FullscreenExitIcon /> : <FullscreenIcon />}
                onClick={() => updateViewState({ viewMaximized: !viewState.viewMaximized })}
              />
            </Tooltip>
          )}
          {shouldEnableArtifactBasedEvaluation() && displaySidebarToggleButton && (
            <Tooltip
              title={
                <FormattedMessage
                  defaultMessage='Toggle the preview sidepane'
                  description='Experiment page > control bar > expanded view toggle button tooltip'
                />
              }
            >
              <ToggleIconButton
                pressed={viewState.previewPaneVisible}
                icon={<SidebarIcon />}
                onClick={() =>
                  updateViewState({ previewPaneVisible: !viewState.previewPaneVisible })
                }
              />
            </Tooltip>
          )}
          <ExperimentViewRefreshButton />
          {/* TODO: Add tooltip to guide users to this button */}
          {shouldEnablePromptLab() && !isComparingExperiments && (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button type='primary' icon={<PlusIcon />}>
                  <FormattedMessage
                    defaultMessage='New run'
                    description='Button used to pop up a modal to create a new run'
                  />
                </Button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Content>
                <DropdownMenu.Item onSelect={() => createNewRun()}>
                  {' '}
                  <FormattedMessage
                    defaultMessage='using Prompt Engineering'
                    description='String for creating a new run with prompt engineering modal'
                  />{' '}
                  <Tag style={{ marginLeft: '4px' }} color='turquoise'>
                    <FormattedMessage
                      defaultMessage='Experimental'
                      description='Experimental badge shown for features which are experimental'
                    />
                  </Tag>
                </DropdownMenu.Item>
                <DropdownMenu.Item onSelect={() => setCreateRunWithNotebookModalOpenValue(true)}>
                  {' '}
                  <FormattedMessage
                    defaultMessage='using Notebook'
                    description='String for creating a new run from a notebook'
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

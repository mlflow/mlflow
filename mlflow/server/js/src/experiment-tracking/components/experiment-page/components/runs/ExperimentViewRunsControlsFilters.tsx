import {
  Button,
  DownloadIcon,
  Dropdown,
  Menu,
  OverflowIcon,
  Select,
  Option,
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useCallback, useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { COLUMN_SORT_BY_ASC, LIFECYCLE_FILTER, SORT_DELIMITER_SYMBOL } from '../../../../constants';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { ExperimentRunSortOption } from '../../hooks/useRunSortOptions';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { TAGS_TO_COLUMNS_MAP } from '../../utils/experimentPage.column-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRefreshButton } from './ExperimentViewRefreshButton';
import { ExperimentViewRunsColumnSelector } from './ExperimentViewRunsColumnSelector';
import { ExperimentViewRunsModeSwitch } from './ExperimentViewRunsModeSwitch';
import { ExperimentViewRunsSortSelector } from './ExperimentViewRunsSortSelector';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';
import { getStartTimeColumnDisplayName } from '../../utils/experimentPage.common-utils';

export type ExperimentViewRunsControlsFiltersProps = {
  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;
  runsData: ExperimentRunsSelectorResult;
  onDownloadCsv: () => void;
  requestError: ErrorWrapper | null;
};

export const ExperimentViewRunsControlsFilters = React.memo(
  ({
    searchFacetsState,
    updateSearchFacets,
    runsData,
    viewState,
    onDownloadCsv,
    requestError,
  }: ExperimentViewRunsControlsFiltersProps) => {
    const { compareRunsMode, startTime, lifecycleFilter } = searchFacetsState;
    const intl = useIntl();

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

    return (
      <>
        <div css={styles.groupBar}>
          <div css={styles.controlBar}>
            <ExperimentViewRunsModeSwitch
              compareRunsMode={compareRunsMode}
              setCompareRunsMode={(newCompareRunsMode) =>
                updateSearchFacets({ compareRunsMode: newCompareRunsMode })
              }
              viewState={viewState}
            />
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
                    onChange={() =>
                      updateSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.ACTIVE })
                    }
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
                    onChange={() =>
                      updateSearchFacets({ lifecycleFilter: LIFECYCLE_FILTER.DELETED })
                    }
                  >
                    <FormattedMessage
                      defaultMessage='Deleted'
                      description='Linked model dropdown option to show deleted experiment runs'
                    />
                  </DialogComboboxOptionListSelectItem>
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
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
            <ExperimentViewRefreshButton />
          </div>
        </div>
      </>
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

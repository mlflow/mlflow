import {
  CopyIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Empty,
  InfoIcon,
  Input,
  SearchIcon,
  Skeleton,
  Spinner,
  ToggleButton,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { EvaluationDataReduxState } from '../../reducers/EvaluationDataReducer';
import { SearchExperimentRunsViewState } from '../experiment-page/models/SearchExperimentRunsViewState';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { EvaluationArtifactCompareTable } from './components/EvaluationArtifactCompareTable';
import { useEvaluationArtifactColumns } from './hooks/useEvaluationArtifactColumns';
import { useEvaluationArtifactTableData } from './hooks/useEvaluationArtifactTableData';
import { useEvaluationArtifactTables } from './hooks/useEvaluationArtifactTables';
import type {
  RunDatasetWithTags,
  UpdateExperimentSearchFacetsFn,
  UpdateExperimentViewStateFn,
} from '../../types';
import { getEvaluationTableArtifact } from '../../actions';
import { FormattedMessage, useIntl } from 'react-intl';
import { PreviewSidebar } from '../../../common/components/PreviewSidebar';
import { useEvaluationArtifactViewState } from './hooks/useEvaluationArtifactViewState';

const MAX_RUNS_TO_COMPARE = 10;

interface EvaluationArtifactCompareViewProps {
  visibleRuns: RunRowType[];
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
}

/**
 * Compares the table data contained in experiment run artifacts.
 */
export const EvaluationArtifactCompareView = ({
  visibleRuns,
  updateSearchFacets,
  onDatasetSelected,
  viewState,
  updateViewState,
}: EvaluationArtifactCompareViewProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const comparedRuns = visibleRuns.slice(0, MAX_RUNS_TO_COMPARE);

  const {
    selectedTables,
    groupByCols,
    outputColumn,
    intersectingOnly,
    setSelectedTables,
    setGroupByCols,
    setOutputColumn,
    setIntersectingOnly,
  } = useEvaluationArtifactViewState(viewState, updateViewState);

  const [showSearchSpinner, setShowSearchSpinner] = useState(false);
  const [filter, setFilter] = useState('');
  const [debouncedFilter, setDebouncedFilter] = useState('');

  const dispatch = useDispatch();

  const handleTableToggle = useCallback(
    (value: string) =>
      setSelectedTables((currentValue) => {
        if (currentValue.includes(value)) {
          return currentValue.filter((item) => item !== value);
        } else {
          return [...currentValue, value];
        }
      }),
    [setSelectedTables],
  );

  const handleGroupByToggle = useCallback(
    (value: string) =>
      setGroupByCols((currentValue) => {
        if (currentValue.includes(value)) {
          return currentValue.filter((item) => item !== value);
        } else {
          return [...currentValue, value];
        }
      }),
    [setGroupByCols],
  );

  const comparedRunsUuids = useMemo(
    () => comparedRuns.map(({ runUuid }) => runUuid),
    [comparedRuns],
  );

  const evaluationArtifactsByRunUuid = useSelector(
    ({ evaluationData }: { evaluationData: EvaluationDataReduxState }) =>
      evaluationData.evaluationArtifactsByRunUuid,
  );

  const { tables, tablesByRun } = useEvaluationArtifactTables(comparedRuns);

  const isLoading = useSelector(
    ({ evaluationData }: { evaluationData: EvaluationDataReduxState }) => {
      return comparedRunsUuids.some((uuid) =>
        selectedTables.some(
          (table) => evaluationData.evaluationArtifactsLoadingByRunUuid[uuid]?.[table],
        ),
      );
    },
  );

  const { columns } = useEvaluationArtifactColumns(
    evaluationArtifactsByRunUuid,
    comparedRunsUuids,
    selectedTables,
  );

  const tableRows = useEvaluationArtifactTableData(
    evaluationArtifactsByRunUuid,
    comparedRunsUuids,
    selectedTables,
    groupByCols,
    outputColumn,
    intersectingOnly,
  );

  // For every run, load its selected tables
  useEffect(() => {
    if (!selectedTables.length) {
      return;
    }
    for (const run of comparedRuns) {
      if (!run) {
        continue;
      }
      const tablesToFetch = (tablesByRun[run.runUuid] || []).filter((table) =>
        selectedTables.includes(table),
      );
      for (const table of tablesToFetch) {
        dispatch(getEvaluationTableArtifact(run.runUuid, table, false));
      }
    }
  }, [comparedRuns, dispatch, selectedTables, tablesByRun]);

  // Table is ready to use if it's loaded, at least one table and at least one run is selected
  const areTablesSelected = selectedTables.length > 0;
  const areRunsSelected = comparedRuns.length > 0;
  const isViewConfigured = !isLoading && areTablesSelected && areRunsSelected;

  const filteredRows = useMemo(() => {
    if (!debouncedFilter.trim()) {
      return tableRows;
    }
    const regexp = new RegExp(debouncedFilter, 'i');
    return tableRows.filter(({ groupByCellValues }) =>
      Object.values(groupByCellValues).some((groupByValue) => groupByValue?.match(regexp)),
    );
  }, [tableRows, debouncedFilter]);

  const handleHideRun = (runUuid: string) =>
    updateSearchFacets((existingFacets) => ({
      ...existingFacets,
      runsHidden: [...existingFacets.runsHidden, runUuid],
    }));

  // Make sure that there's at least one "group by" column selected
  useEffect(() => {
    const noColumnsSelected = groupByCols.length < 1;
    const columnNotAvailableAnymore = groupByCols.some((column) => !columns.includes(column));
    const firstAvailableColumn = columns[0];

    if ((noColumnsSelected || columnNotAvailableAnymore) && firstAvailableColumn) {
      setGroupByCols([firstAvailableColumn]);
    }
  }, [groupByCols, outputColumn, columns, setGroupByCols]);

  // All columns that are not used for grouping can be used as output (compare) column
  const availableOutputColumns = useMemo(
    () => columns.filter((col) => !groupByCols.includes(col)),
    [columns, groupByCols],
  );

  // If the current output column have been selected as "group by", change it to the other available one
  useEffect(() => {
    if (groupByCols.includes(outputColumn) || !outputColumn) {
      setOutputColumn(availableOutputColumns[0] || '');
    }
  }, [groupByCols, outputColumn, availableOutputColumns, setOutputColumn]);

  // On every change of "filter" input, debounce the value and show the spinner
  useEffect(() => {
    setShowSearchSpinner(true);
    const handler = setTimeout(() => setDebouncedFilter(filter), 250);
    return () => clearTimeout(handler);
  }, [filter]);

  // After debounced filter value settles in, remove the search spinner
  useEffect(() => {
    setShowSearchSpinner(false);
  }, [debouncedFilter]);

  // If the current output column is not available anymore, change it to the other available one
  useEffect(() => {
    if (!availableOutputColumns.includes(outputColumn)) {
      setOutputColumn(availableOutputColumns[0] || '');
    }
  }, [outputColumn, availableOutputColumns, setOutputColumn]);

  // If any currently selected table is not available anymore, deselect it
  useEffect(() => {
    if (selectedTables.some((table) => !tables.includes(table))) {
      setSelectedTables([]);
    }
  }, [selectedTables, tables, setSelectedTables]);

  // Disable intersection mode when changing compared runs
  useEffect(() => {
    setIntersectingOnly(false);
  }, [comparedRuns, setIntersectingOnly]);

  const [sidebarPreviewData, setSidebarPreviewData] = useState<{
    value: string;
    header: string;
  } | null>(null);

  const handleCellClicked = (value: string, header: string) => {
    setSidebarPreviewData({ value, header });
    updateViewState({ previewPaneVisible: true });
  };

  return (
    <div
      css={{
        borderTop: `1px solid ${theme.colors.borderDecorative}`,
        height: '100%',
        display: 'grid',
        gridTemplateColumns: viewState.previewPaneVisible ? '1fr auto' : '1fr',
        columnGap: theme.spacing.sm,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          paddingLeft: theme.spacing.sm,
          paddingTop: theme.spacing.sm,
          height: '100%',
          display: 'grid',
          gridTemplateRows: 'auto auto 1fr',
          overflow: 'hidden',
          rowGap: theme.spacing.sm,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            overflow: 'hidden',
            height: theme.general.heightSm,
          }}
        >
          <DialogCombobox
            label={
              <FormattedMessage
                defaultMessage='Table'
                description='Experiment page > artifact compare view > table select dropdown label'
              />
            }
            multiSelect
            value={selectedTables}
          >
            <DialogComboboxTrigger
              css={{ maxWidth: 300 }}
              data-testid='dropdown-tables'
              onClear={() => setSelectedTables([])}
            />
            <DialogComboboxContent css={{ maxWidth: 300 }}>
              <DialogComboboxOptionList>
                {tables.map((artifactPath) => (
                  <DialogComboboxOptionListCheckboxItem
                    value={artifactPath}
                    key={artifactPath}
                    onChange={handleTableToggle}
                    checked={selectedTables.includes(artifactPath)}
                    data-testid='dropdown-tables-option'
                  >
                    {artifactPath}
                  </DialogComboboxOptionListCheckboxItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
          <Tooltip
            title={
              <FormattedMessage
                defaultMessage='Using the list of logged table artifacts, select at least one to start comparing results.'
                description='Experiment page > artifact compare view > table select dropdown tooltip'
              />
            }
          >
            <InfoIcon />
          </Tooltip>
        </div>
        {isLoading ? (
          <Skeleton />
        ) : (
          <>
            <div
              css={{
                display: 'flex',
                columnGap: theme.spacing.sm,
                alignItems: 'center',
                overflow: 'hidden',
                height: theme.general.heightSm,
              }}
            >
              <Input
                prefix={<SearchIcon />}
                suffix={showSearchSpinner && <Spinner size='small' />}
                css={{ width: 300, minWidth: 300 }}
                onChange={(e) => setFilter(e.target.value)}
                value={filter}
                placeholder={intl.formatMessage(
                  {
                    defaultMessage: 'Filter by {columnNames}',
                    description:
                      'Experiment page > artifact compare view > search input placeholder',
                  },
                  {
                    columnNames: groupByCols.join(', '),
                  },
                )}
                allowClear
                disabled={!isViewConfigured}
              />
              <DialogCombobox
                value={groupByCols}
                multiSelect
                label={
                  <FormattedMessage
                    defaultMessage='Group by'
                    description='Experiment page > artifact compare view > "group by column" select dropdown label'
                  />
                }
              >
                <DialogComboboxTrigger
                  disabled={!isViewConfigured}
                  allowClear={false}
                  showTagAfterValueCount={1}
                  css={{ maxWidth: 300 }}
                />
                <DialogComboboxContent css={{ maxWidth: 300 }}>
                  <DialogComboboxOptionList>
                    {columns.map((columnName) => (
                      <DialogComboboxOptionListCheckboxItem
                        value={columnName}
                        key={columnName}
                        onChange={handleGroupByToggle}
                        checked={groupByCols.includes(columnName)}
                      >
                        {columnName}
                      </DialogComboboxOptionListCheckboxItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
              <DialogCombobox
                value={[outputColumn]}
                label={
                  <FormattedMessage
                    defaultMessage='Compare'
                    description='Experiment page > artifact compare view > "compare" select dropdown label'
                  />
                }
              >
                <DialogComboboxTrigger
                  disabled={!isViewConfigured}
                  allowClear={false}
                  css={{ maxWidth: 300 }}
                />
                <DialogComboboxContent css={{ maxWidth: 300 }}>
                  <DialogComboboxOptionList>
                    {availableOutputColumns.map((columnName) => (
                      <DialogComboboxOptionListSelectItem
                        value={columnName}
                        key={columnName}
                        onChange={() => setOutputColumn(columnName)}
                        checked={outputColumn === columnName}
                      >
                        {columnName}
                      </DialogComboboxOptionListSelectItem>
                    ))}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
              {/* Disabling this for now, as it's not working as expected and we need to sync with OSS */}
              {false && (
                <Tooltip
                  title={
                    <FormattedMessage
                      defaultMessage='Only show rows where the "compare" column has a value for every run'
                      description='Experiment page > artifact compare view > intersection only toggle checkbox tooltip'
                    />
                  }
                >
                  <ToggleButton
                    disabled={!isViewConfigured}
                    pressed={intersectingOnly}
                    onPressedChange={setIntersectingOnly}
                  >
                    <FormattedMessage
                      defaultMessage='Show intersection only'
                      description='Experiment page > artifact compare view > intersection only toggle checkbox'
                    />
                  </ToggleButton>
                </Tooltip>
              )}
            </div>

            {isViewConfigured ? (
              <div
                css={{
                  position: 'relative' as const,
                  zIndex: 1,
                  overflowY: 'hidden' as const,
                  height: '100%',
                }}
              >
                <EvaluationArtifactCompareTable
                  comparedRuns={comparedRuns}
                  groupByColumns={groupByCols}
                  resultList={filteredRows}
                  onCellClick={handleCellClicked}
                  onHideRun={handleHideRun}
                  onDatasetSelected={onDatasetSelected}
                  highlightedText={debouncedFilter.trim()}
                />
              </div>
            ) : (
              <div css={{ marginTop: theme.spacing.lg }}>
                <Empty
                  title={
                    areRunsSelected ? (
                      <FormattedMessage
                        defaultMessage='No tables selected'
                        description='Experiment page > artifact compare view > empty state for no tables selected > title'
                      />
                    ) : (
                      <FormattedMessage
                        defaultMessage='No runs selected'
                        description='Experiment page > artifact compare view > empty state for no runs selected > title'
                      />
                    )
                  }
                  description={
                    areRunsSelected ? (
                      <FormattedMessage
                        defaultMessage='Using controls above, select at least one artifact containing table.'
                        description='Experiment page > artifact compare view > empty state for no tables selected > subtitle with the hint'
                      />
                    ) : (
                      <FormattedMessage
                        defaultMessage='Make sure that at least one experiment run is visible and available to compare'
                        description='Experiment page > artifact compare view > empty state for no runs selected > subtitle with the hint'
                      />
                    )
                  }
                />
              </div>
            )}
          </>
        )}
      </div>
      {viewState.previewPaneVisible && (
        <PreviewSidebar
          content={sidebarPreviewData?.value}
          copyText={sidebarPreviewData?.value}
          headerText={sidebarPreviewData?.header}
          empty={
            <Empty
              description={
                <FormattedMessage
                  defaultMessage='Select a cell to display preview'
                  description='Experiment page > artifact compare view > preview sidebar > nothing selected'
                />
              }
            />
          }
        />
      )}
    </div>
  );
};

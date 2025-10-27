import {
  CopyIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Empty,
  InfoSmallIcon,
  Input,
  SearchIcon,
  LegacySkeleton,
  Spinner,
  ToggleButton,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { EvaluationDataReduxState } from '../../reducers/EvaluationDataReducer';
import type { ExperimentPageViewState } from '../experiment-page/models/ExperimentPageViewState';
import type { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { EvaluationArtifactCompareTable } from './components/EvaluationArtifactCompareTable';
import { useEvaluationArtifactColumns } from './hooks/useEvaluationArtifactColumns';
import { useEvaluationArtifactTableData } from './hooks/useEvaluationArtifactTableData';
import { useEvaluationArtifactTables } from './hooks/useEvaluationArtifactTables';
import type { RunDatasetWithTags, UpdateExperimentViewStateFn } from '../../types';
import { FormattedMessage, useIntl } from 'react-intl';
import { PreviewSidebar } from '../../../common/components/PreviewSidebar';
import { useEvaluationArtifactViewState } from './hooks/useEvaluationArtifactViewState';
import { useEvaluationArtifactWriteBack } from './hooks/useEvaluationArtifactWriteBack';
import { PromptEngineeringContextProvider } from './contexts/PromptEngineeringContext';
import type { ReduxState, ThunkDispatch } from '../../../redux-types';
import { getEvaluationTableArtifact } from '../../actions';
import Utils from '../../../common/utils/Utils';
import {
  DEFAULT_PROMPTLAB_OUTPUT_COLUMN,
  canEvaluateOnRun,
  extractRequiredInputParamsForRun,
} from '../prompt-engineering/PromptEngineering.utils';
import { searchAllPromptLabAvailableEndpoints } from '../../actions/PromptEngineeringActions';
import { shouldEnablePromptLab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import {
  EvaluationArtifactViewEmptyState,
  shouldDisplayEvaluationArtifactEmptyState,
} from './EvaluationArtifactViewEmptyState';
import { useUpdateExperimentViewUIState } from '../experiment-page/contexts/ExperimentPageUIStateContext';
import { useToggleRowVisibilityCallback } from '../experiment-page/hooks/useToggleRowVisibilityCallback';
import { RUNS_VISIBILITY_MODE } from '../experiment-page/models/ExperimentPageUIState';
import { FormattedJsonDisplay } from '@mlflow/mlflow/src/common/components/JsonFormatting';
import { EvaluationTableParseError } from '../../sdk/EvaluationArtifactService';

const MAX_RUNS_TO_COMPARE = 10;

interface EvaluationArtifactCompareViewProps {
  comparedRuns: RunRowType[];
  viewState: ExperimentPageViewState;
  updateViewState: UpdateExperimentViewStateFn;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
}

/**
 * Compares the table data contained in experiment run artifacts.
 */
const EvaluationArtifactCompareViewImpl = ({
  comparedRuns,
  onDatasetSelected,
  viewState,
  updateViewState,
}: EvaluationArtifactCompareViewProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const updateUIState = useUpdateExperimentViewUIState();

  const visibleRuns = useMemo(
    () => comparedRuns.filter(({ hidden }) => !hidden).slice(0, MAX_RUNS_TO_COMPARE),
    [comparedRuns],
  );

  const { selectedTables, groupByCols, outputColumn, setSelectedTables, setGroupByCols, setOutputColumn } =
    useEvaluationArtifactViewState(viewState, updateViewState);

  const [showSearchSpinner, setShowSearchSpinner] = useState(false);
  const [filter, setFilter] = useState('');
  const [debouncedFilter, setDebouncedFilter] = useState('');
  const [userDeselectedAllColumns, setUserDeselectedAllColumns] = useState(false);

  const { isSyncingArtifacts, EvaluationSyncStatusElement } = useEvaluationArtifactWriteBack();

  const dispatch = useDispatch<ThunkDispatch>();

  useEffect(() => {
    if (shouldEnablePromptLab()) {
      dispatch(searchAllPromptLabAvailableEndpoints()).catch((e) => {
        Utils.logErrorAndNotifyUser(e?.message || e);
      });
    }
  }, [dispatch]);

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
        const newValues = currentValue.includes(value)
          ? currentValue.filter((item) => item !== value)
          : [...currentValue, value];
        setUserDeselectedAllColumns(newValues.length === 0);
        return newValues;
      }),
    [setGroupByCols],
  );

  const visibleRunsUuids = useMemo(() => visibleRuns.map(({ runUuid }) => runUuid), [visibleRuns]);

  const { evaluationArtifactsByRunUuid, evaluationPendingDataByRunUuid, evaluationDraftInputValues } = useSelector(
    ({ evaluationData }: ReduxState) => evaluationData,
  );

  const { tables, tablesByRun, noEvalTablesLogged } = useEvaluationArtifactTables(visibleRuns);

  // Select the first table by default
  useEffect(() => {
    if (tables.length > 0 && selectedTables.length === 0) {
      setSelectedTables([tables[0]]);
    }
  }, [tables, setSelectedTables, selectedTables.length]);

  const isLoading = useSelector(({ evaluationData, modelGateway }: ReduxState) => {
    const gatewayRoutesLoading = modelGateway.modelGatewayRoutesLoading.loading;
    return (
      gatewayRoutesLoading ||
      visibleRunsUuids.some((uuid) =>
        selectedTables.some((table) => evaluationData.evaluationArtifactsLoadingByRunUuid[uuid]?.[table]),
      )
    );
  });

  const { columns, imageColumns } = useEvaluationArtifactColumns(
    evaluationArtifactsByRunUuid,
    visibleRunsUuids,
    selectedTables,
  );

  const isImageColumn = imageColumns.includes(outputColumn);

  const tableRows = useEvaluationArtifactTableData(
    evaluationArtifactsByRunUuid,
    evaluationPendingDataByRunUuid,
    evaluationDraftInputValues,
    visibleRunsUuids,
    selectedTables,
    groupByCols,
    outputColumn,
  );

  // Try to extract all existing prompt input fields from prompt engineering runs, if there are any.
  // Return "null" otherwise.
  const promptLabInputVariableNames = useMemo(() => {
    const promptEngineeringRuns = visibleRuns.filter(canEvaluateOnRun);
    const allInputNames = promptEngineeringRuns.map(extractRequiredInputParamsForRun).flat();
    if (!allInputNames.length) {
      return null;
    }

    // Remove duplicates
    const distinctInputNames = Array.from(new Set(allInputNames));

    // Ensure that detected input names are included in the available columns
    return distinctInputNames.filter((inputName) => columns.includes(inputName));
  }, [visibleRuns, columns]);

  // If we've changed the visible run set and all of them originate from prompt engineering,
  // reset the columns so they will be recalculated again
  useEffect(() => {
    if (visibleRuns.every(canEvaluateOnRun)) {
      setGroupByCols([]);
    }
  }, [setGroupByCols, visibleRuns]);

  // For every run, load its selected tables
  useEffect(() => {
    if (!selectedTables.length) {
      return;
    }
    for (const run of visibleRuns) {
      if (!run) {
        continue;
      }
      const tablesToFetch = (tablesByRun[run.runUuid] || []).filter((table) => selectedTables.includes(table));
      for (const table of tablesToFetch) {
        dispatch(getEvaluationTableArtifact(run.runUuid, table, false)).catch((e) => {
          if (e instanceof EvaluationTableParseError) {
            // In case of table parse errors, just display the error to the user without propagating it upstream
            Utils.displayGlobalErrorNotification(e.message);
          } else {
            Utils.logErrorAndNotifyUser(e.message || e);
          }
        });
      }
    }
  }, [visibleRuns, dispatch, selectedTables, tablesByRun]);

  // Table is ready to use if it's loaded, at least one table and at least one run is selected
  const areTablesSelected = selectedTables.length > 0;
  const areRunsSelected = visibleRuns.length > 0;
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

  const toggleRowVisibility = useToggleRowVisibilityCallback(comparedRuns);

  const handleHideRun = useCallback(
    (runUuid: string) => {
      toggleRowVisibility(RUNS_VISIBILITY_MODE.CUSTOM, runUuid);
    },
    [toggleRowVisibility],
  );

  // Make sure that there's at least one "group by" column selected
  useEffect(() => {
    if (isLoading || userDeselectedAllColumns) {
      return;
    }
    const noColumnsSelected = groupByCols.length < 1;
    const columnNotAvailableAnymore = groupByCols.some((column) => !columns.includes(column));
    const firstColumn = columns[0];

    // If prompt engineering prompt inputs are detected, take them as a candidate for initial "group by" columns.
    // If not, use the first valid column found.
    const groupByColumnCandidates = promptLabInputVariableNames || (firstColumn ? [firstColumn] : null);

    if ((noColumnsSelected || columnNotAvailableAnymore) && groupByColumnCandidates) {
      setGroupByCols(groupByColumnCandidates);
    }
  }, [
    isLoading,
    userDeselectedAllColumns,
    groupByCols,
    outputColumn,
    columns,
    setGroupByCols,
    promptLabInputVariableNames,
  ]);

  // Remove MLFLOW_ columns from the list of groupby columns since they are for metadata only
  const availableGroupByColumns = useMemo(() => columns.filter((col) => !col.startsWith('MLFLOW_')), [columns]);

  // All columns that are not used for grouping can be used as output (compare) column
  // Remove MLFLOW_ columns from the list of output columns
  const availableOutputColumns = useMemo(
    () => [...columns, ...imageColumns].filter((col) => !groupByCols.includes(col) && !col.startsWith('MLFLOW_')),
    [columns, imageColumns, groupByCols],
  );

  // If the current output column have been selected as "group by", change it to the other available one
  useEffect(() => {
    if (groupByCols.includes(outputColumn) || !outputColumn) {
      const nextColumnCandidate = availableOutputColumns.includes(DEFAULT_PROMPTLAB_OUTPUT_COLUMN)
        ? DEFAULT_PROMPTLAB_OUTPUT_COLUMN
        : availableOutputColumns[0];
      setOutputColumn(nextColumnCandidate || '');
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
      const nextColumnCandidate = availableOutputColumns.includes(DEFAULT_PROMPTLAB_OUTPUT_COLUMN)
        ? DEFAULT_PROMPTLAB_OUTPUT_COLUMN
        : availableOutputColumns[0];
      setOutputColumn(nextColumnCandidate || '');
    }
  }, [outputColumn, availableOutputColumns, setOutputColumn]);

  // If any currently selected table is not available anymore, deselect it
  useEffect(() => {
    if (selectedTables.some((table) => !tables.includes(table))) {
      setSelectedTables([]);
    }
  }, [selectedTables, tables, setSelectedTables]);

  const [sidebarPreviewData, setSidebarPreviewData] = useState<{
    value: string;
    header: string;
  } | null>(null);

  const handleCellClicked = useCallback(
    (value: string, header: string) => {
      setSidebarPreviewData({ value, header });
      updateViewState({ previewPaneVisible: true });
    },
    [updateViewState],
  );

  return (
    <div
      css={{
        flex: 1,
        borderTop: `1px solid ${theme.colors.border}`,
        borderLeft: `1px solid ${theme.colors.border}`,
        // Let's cover 1 pixel of the grid's border for the sleek look
        marginLeft: -1,
        zIndex: 1,
        height: '100%',
        display: 'grid',
        gridTemplateColumns: viewState.previewPaneVisible ? '1fr auto' : '1fr',
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
          backgroundColor: theme.colors.backgroundSecondary,
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
            componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_358"
            label={
              <FormattedMessage
                defaultMessage="Table"
                description="Experiment page > artifact compare view > table select dropdown label"
              />
            }
            multiSelect
            value={selectedTables}
          >
            <DialogComboboxTrigger
              css={{ maxWidth: 300, backgroundColor: theme.colors.backgroundPrimary }}
              data-testid="dropdown-tables"
              onClear={() => setSelectedTables([])}
              disabled={isSyncingArtifacts || !areRunsSelected || noEvalTablesLogged}
            />
            <DialogComboboxContent css={{ maxWidth: 300 }}>
              <DialogComboboxOptionList>
                {tables.map((artifactPath) => (
                  <DialogComboboxOptionListCheckboxItem
                    value={artifactPath}
                    key={artifactPath}
                    onChange={handleTableToggle}
                    checked={selectedTables.includes(artifactPath)}
                    data-testid="dropdown-tables-option"
                  >
                    {artifactPath}
                  </DialogComboboxOptionListCheckboxItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
          <LegacyTooltip
            title={
              <FormattedMessage
                defaultMessage="Using the list of logged table artifacts, select at least one to start comparing results."
                description="Experiment page > artifact compare view > table select dropdown tooltip"
              />
            }
          >
            <InfoSmallIcon />
          </LegacyTooltip>
        </div>
        {isLoading ? (
          <LegacySkeleton />
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
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_414"
                prefix={<SearchIcon />}
                suffix={showSearchSpinner && <Spinner size="small" />}
                css={{ width: 300, minWidth: 300 }}
                onChange={(e) => setFilter(e.target.value)}
                value={filter}
                placeholder={intl.formatMessage(
                  {
                    defaultMessage: 'Filter by {columnNames}',
                    description: 'Experiment page > artifact compare view > search input placeholder',
                  },
                  {
                    columnNames: groupByCols.join(', '),
                  },
                )}
                allowClear
                disabled={!isViewConfigured || isSyncingArtifacts}
              />
              <DialogCombobox
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_433"
                value={groupByCols}
                multiSelect
                label={
                  <FormattedMessage
                    defaultMessage="Group by"
                    description='Experiment page > artifact compare view > "group by column" select dropdown label'
                  />
                }
              >
                <DialogComboboxTrigger
                  disabled={!isViewConfigured || isSyncingArtifacts}
                  allowClear={false}
                  showTagAfterValueCount={1}
                  css={{ maxWidth: 300, backgroundColor: theme.colors.backgroundPrimary }}
                  aria-label='Select "group by" columns'
                />
                <DialogComboboxContent css={{ maxWidth: 300 }}>
                  <DialogComboboxOptionList>
                    {availableGroupByColumns.map((columnName) => (
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
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationartifactcompareview.tsx_465"
                value={[outputColumn]}
                label={
                  <FormattedMessage
                    defaultMessage="Compare"
                    description='Experiment page > artifact compare view > "compare" select dropdown label'
                  />
                }
              >
                <DialogComboboxTrigger
                  disabled={!isViewConfigured || isSyncingArtifacts}
                  allowClear={false}
                  css={{ maxWidth: 300, backgroundColor: theme.colors.backgroundPrimary }}
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
            </div>

            {shouldDisplayEvaluationArtifactEmptyState({
              areRunsSelected,
              areTablesSelected,
              noEvalTablesLogged,
              userDeselectedAllColumns,
            }) ? (
              <div
                css={{
                  height: '100%',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <EvaluationArtifactViewEmptyState
                  areRunsSelected={areRunsSelected}
                  areTablesSelected={areTablesSelected}
                  noEvalTablesLogged={noEvalTablesLogged}
                  userDeselectedAllColumns={userDeselectedAllColumns}
                />
              </div>
            ) : (
              <div
                css={{
                  position: 'relative' as const,
                  zIndex: 1,
                  overflowY: 'hidden' as const,
                  height: '100%',
                  backgroundColor: theme.colors.backgroundPrimary,
                }}
              >
                <PromptEngineeringContextProvider tableData={tableRows} outputColumn={outputColumn}>
                  <EvaluationArtifactCompareTable
                    visibleRuns={visibleRuns}
                    groupByColumns={groupByCols}
                    resultList={filteredRows}
                    onCellClick={isImageColumn ? undefined : handleCellClicked}
                    onHideRun={handleHideRun}
                    onDatasetSelected={onDatasetSelected}
                    highlightedText={debouncedFilter.trim()}
                    isPreviewPaneVisible={viewState.previewPaneVisible}
                    outputColumnName={outputColumn}
                    isImageColumn={isImageColumn}
                  />
                </PromptEngineeringContextProvider>
              </div>
            )}
            {EvaluationSyncStatusElement}
          </>
        )}
      </div>
      {viewState.previewPaneVisible && (
        <PreviewSidebar
          content={sidebarPreviewData?.value ? <FormattedJsonDisplay json={sidebarPreviewData.value} /> : null}
          copyText={sidebarPreviewData?.value || ''}
          headerText={sidebarPreviewData?.header}
          onClose={() => updateViewState({ previewPaneVisible: false })}
          empty={
            <Empty
              description={
                <FormattedMessage
                  defaultMessage="Select a cell to display preview"
                  description="Experiment page > artifact compare view > preview sidebar > nothing selected"
                />
              }
            />
          }
        />
      )}
    </div>
  );
};

export const EvaluationArtifactCompareView = (props: EvaluationArtifactCompareViewProps & { disabled?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  if (props.disabled) {
    return (
      <div
        css={{
          flex: 1,
          backgroundColor: theme.colors.backgroundSecondary,
          height: '100%',
          borderTop: `1px solid ${theme.colors.border}`,
          borderLeft: `1px solid ${theme.colors.border}`,
          paddingTop: theme.spacing.lg,
          marginLeft: -1,
          zIndex: 1,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Empty
          title={
            <FormattedMessage
              defaultMessage="Evaluation not available when grouping is enabled"
              description="Experiment page > artifact compare view > disabled due to run grouping > title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Disable run grouping in order to access the evaluation view"
              description="Experiment page > artifact compare view > disabled due to run grouping > description"
            />
          }
          image={<div />}
        />
      </div>
    );
  }
  return <EvaluationArtifactCompareViewImpl {...props} />;
};

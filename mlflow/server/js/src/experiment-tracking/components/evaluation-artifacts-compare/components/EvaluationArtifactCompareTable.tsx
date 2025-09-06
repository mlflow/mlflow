import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import type { UseEvaluationArtifactTableDataResult } from '../hooks/useEvaluationArtifactTableData';
import type { CellClickedEvent, ColDef, GridApi } from '@ag-grid-community/core';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { MLFlowAgGridLoader } from '../../../../common/components/ag-grid/AgGridLoader';
import { EvaluationRunHeaderCellRenderer } from './EvaluationRunHeaderCellRenderer';
import { EvaluationTextCellRenderer } from './EvaluationTextCellRenderer';
import {
  EVALUATION_ARTIFACTS_TABLE_ROW_HEIGHT,
  EVALUATION_ARTIFACTS_TEXT_COLUMN_WIDTH,
  getEvaluationArtifactsTableHeaderHeight,
} from '../EvaluationArtifactCompare.utils';
import { EvaluationGroupByHeaderCellRenderer } from './EvaluationGroupByHeaderCellRenderer';
import type { Theme } from '@emotion/react';
import type { RunDatasetWithTags } from '../../../types';
import { useEvaluationAddNewInputsModal } from '../hooks/useEvaluationAddNewInputsModal';
import { useDispatch, useSelector } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { evaluateAddInputValues } from '../../../actions/PromptEngineeringActions';
import { canEvaluateOnRun, extractRequiredInputParamsForRun } from '../../prompt-engineering/PromptEngineering.utils';
import { useIntl } from 'react-intl';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';
import { EvaluationTableHeader } from './EvaluationTableHeader';
import { EvaluationTableActionsColumnRenderer } from './EvaluationTableActionsColumnRenderer';
import { EvaluationTableActionsCellRenderer } from './EvaluationTableActionsCellRenderer';
import { shouldEnablePromptLab } from '../../../../common/utils/FeatureUtils';
import { useCreateNewRun } from '../../experiment-page/hooks/useCreateNewRun';
import { EvaluationImageCellRenderer } from './EvaluationImageCellRenderer';

export interface EvaluationArtifactCompareTableProps {
  resultList: UseEvaluationArtifactTableDataResult;
  visibleRuns: RunRowType[];
  groupByColumns: string[];
  onCellClick?: (value: string, columnHeader: string) => void;
  onHideRun: (runUuid: string) => void;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
  highlightedText: string;
  isPreviewPaneVisible?: boolean;
  outputColumnName: string;
  isImageColumn: boolean;
}

export const EvaluationArtifactCompareTable = ({
  resultList,
  visibleRuns,
  groupByColumns,
  onCellClick,
  onHideRun,
  onDatasetSelected,
  highlightedText = '',
  isPreviewPaneVisible,
  outputColumnName,
  isImageColumn,
}: EvaluationArtifactCompareTableProps) => {
  const [columns, setColumns] = useState<ColDef[]>([]);

  const [gridApi, setGridApi] = useState<GridApi | null>(null);
  const pendingData = useSelector(({ evaluationData }: ReduxState) => evaluationData.evaluationPendingDataByRunUuid);
  const gridWrapperRef = useRef<HTMLDivElement>(null);

  const { isHeaderExpanded } = usePromptEngineeringContext();
  const { createNewRun } = useCreateNewRun();

  // Before hiding or duplicating the run, let's refresh the header to mitigate ag-grid's
  // bug where it fails to defocus cell after the whole table has been hidden.
  const handleHideRun = useCallback(
    (runUuid: string) => {
      gridApi?.refreshHeader();
      onHideRun(runUuid);
    },
    [gridApi, onHideRun],
  );

  const handleDuplicateRun = useCallback(
    (runToDuplicate?: RunRowType) => {
      gridApi?.refreshHeader();
      createNewRun(runToDuplicate);
    },
    [createNewRun, gridApi],
  );

  useEffect(() => {
    if (gridApi && !isPreviewPaneVisible) {
      gridApi.clearFocusedCell();
    }
  }, [gridApi, isPreviewPaneVisible]);

  // Force-refresh visible cells' values when some pending data have changes
  // either by loading new data or discarding values. This makes sure
  // that even if the prompt evaluates to the same value, the grid still refreshes.
  useEffect(() => {
    if (!gridApi) {
      return;
    }
    const visibleRows = gridApi.getRenderedNodes();
    gridApi.refreshCells({ force: true, rowNodes: visibleRows });
  }, [gridApi, pendingData, highlightedText]);

  const { showAddNewInputsModal, AddNewInputsModal } = useEvaluationAddNewInputsModal();
  const dispatch = useDispatch<ThunkDispatch>();

  const scrollGridToTop = useCallback(() => {
    // Find the scrollable viewport element
    const gridViewport = gridWrapperRef.current?.querySelector('.ag-body-viewport');
    if (gridViewport) {
      gridViewport.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      // If for some reason there's no element, use native jumpy method
      gridApi?.ensureIndexVisible(0, 'top');
    }
  }, [gridApi]);

  const displayAddNewInputsButton = useMemo(
    // TODO(ML-32969): count prompt-engineered runs based on tags
    () => visibleRuns.map(extractRequiredInputParamsForRun).flat().length > 0,
    [visibleRuns],
  );

  const onAddNewInputs = useCallback(() => {
    showAddNewInputsModal(visibleRuns, (values) => {
      dispatch(evaluateAddInputValues(values));
      // Scroll the grid to the top after adding new row
      scrollGridToTop();
    });
  }, [scrollGridToTop, showAddNewInputsModal, dispatch, visibleRuns]);

  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const handleCellClicked = useCallback(
    ({ value, colDef, column }: CellClickedEvent) => {
      const emptyMessage = intl.formatMessage({
        defaultMessage: '(empty)',
        description: 'Experiment page > artifact compare view > results table > no result (empty cell)',
      });
      return onCellClick?.(value || emptyMessage, colDef.headerName || column.getId());
    },
    [intl, onCellClick],
  );

  const outputColumnIndicator = useMemo(
    () => <Typography.Text bold>{outputColumnName}</Typography.Text>,
    [outputColumnName],
  );

  useEffect(() => {
    const cols: ColDef[] = [];

    const { initialWidthGroupBy, initialWidthOutput, maxWidth, minWidth } = EVALUATION_ARTIFACTS_TEXT_COLUMN_WIDTH;

    if (shouldEnablePromptLab() && visibleRuns.some((run) => canEvaluateOnRun(run))) {
      cols.push({
        resizable: false,
        pinned: true,
        width: 40,
        headerComponent: 'ActionsColumnRenderer',
        cellRendererSelector: ({ rowIndex }) =>
          rowIndex === 0
            ? {
                component: 'ActionsCellRenderer',
                params: {
                  displayAddNewInputsButton,
                  onAddNewInputs,
                },
              }
            : undefined,
        cellClass: 'leading-column-cell',
      });
    }

    groupByColumns.forEach((col, index) => {
      const isLastGroupByColumns = index === groupByColumns.length - 1;
      cols.push({
        resizable: true,
        initialWidth: initialWidthGroupBy,
        minWidth,
        maxWidth,
        headerName: col,
        valueGetter: ({ data }) => data.groupByCellValues[col],
        suppressMovable: true,
        cellRenderer: 'TextRendererCellRenderer',
        headerClass: isLastGroupByColumns ? 'last-group-by-header-cell' : undefined,
        cellRendererParams: {
          isGroupByColumn: true,
        },
        headerComponent: 'GroupHeaderCellRenderer',
        headerComponentParams: {
          displayAddNewInputsButton,
          onAddNewInputs,
        },
        colId: col,
        onCellClicked: handleCellClicked,
      });
    });

    visibleRuns.forEach((run, index) => {
      const isFirstColumn = index === 0;
      cols.push({
        resizable: true,
        initialWidth: initialWidthOutput,
        minWidth,
        maxWidth,
        headerName: run.runName,
        colId: run.runUuid,
        valueGetter: ({ data }) => data.cellValues[run.runUuid],
        suppressMovable: true,
        cellRenderer: isImageColumn ? 'ImageRendererCellRenderer' : 'TextRendererCellRenderer',
        cellRendererParams: {
          run,
        },
        headerComponent: 'RunHeaderCellRenderer',
        headerComponentParams: {
          run,
          onDuplicateRun: handleDuplicateRun,
          onHideRun: handleHideRun,
          onDatasetSelected,
          groupHeaderContent: isFirstColumn ? outputColumnIndicator : null,
        },
        onCellClicked: handleCellClicked,
      });
    });

    setColumns(cols);
  }, [
    visibleRuns,
    groupByColumns,
    handleHideRun,
    handleDuplicateRun,
    onDatasetSelected,
    onAddNewInputs,
    displayAddNewInputsButton,
    handleCellClicked,
    outputColumnIndicator,
    isImageColumn,
  ]);

  useEffect(() => {
    if (!gridApi) {
      return;
    }

    // Check if we need to have a tall header, i.e. if we have any runs
    // with datasets or with evaluation metadata
    const runsContainHeaderMetadata = visibleRuns.some((run) => canEvaluateOnRun(run) || run.datasets?.length > 0);

    // Set header height dynamically
    gridApi.setHeaderHeight(getEvaluationArtifactsTableHeaderHeight(isHeaderExpanded, runsContainHeaderMetadata));
  }, [gridApi, isHeaderExpanded, visibleRuns]);

  return (
    <div css={{ height: '100%', overflow: 'hidden' }} ref={gridWrapperRef}>
      <MLFlowAgGridLoader
        css={createTableStyles(theme)}
        context={{ highlightedText }}
        rowHeight={EVALUATION_ARTIFACTS_TABLE_ROW_HEIGHT}
        onGridReady={({ api }) => setGridApi(api)}
        getRowId={({ data }) => data.key}
        suppressHorizontalScroll={false}
        columnDefs={columns}
        rowData={resultList}
        components={{
          TextRendererCellRenderer: EvaluationTextCellRenderer,
          GroupHeaderCellRenderer: EvaluationGroupByHeaderCellRenderer,
          RunHeaderCellRenderer: EvaluationRunHeaderCellRenderer,
          ActionsColumnRenderer: EvaluationTableActionsColumnRenderer,
          ActionsCellRenderer: EvaluationTableActionsCellRenderer,
          ImageRendererCellRenderer: EvaluationImageCellRenderer,
        }}
      />
      {AddNewInputsModal}
    </div>
  );
};

const createTableStyles = (theme: Theme) => ({
  '.ag-row:not(.ag-row-first), .ag-body-viewport': {
    borderTop: `1px solid ${theme.colors.borderDecorative}`,
  },
  '.ag-row-last': {
    borderBottom: `1px solid ${theme.colors.borderDecorative}`,
  },
  '.ag-cell, .last-group-by-header-cell .header-group-cell': {
    borderRight: `1px solid ${theme.colors.borderDecorative}`,
  },
  '.ag-cell-focus:not(.leading-column-cell)::after': {
    content: '""',
    position: 'absolute' as const,
    inset: 0,
    boxShadow: `inset 0 0 0px 2px ${theme.colors.blue300}`,
    pointerEvents: 'none' as const,
  },
});

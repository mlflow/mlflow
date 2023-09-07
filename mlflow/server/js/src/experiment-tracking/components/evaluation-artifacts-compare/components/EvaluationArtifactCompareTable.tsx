import { useEffect, useMemo, useState } from 'react';
import { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { UseEvaluationArtifactTableDataResult } from '../hooks/useEvaluationArtifactTableData';
import { CellClickedEvent, ColDef } from '@ag-grid-community/core';
import { useDesignSystemTheme } from '@databricks/design-system';
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

export interface EvaluationArtifactCompareTableProps {
  resultList: UseEvaluationArtifactTableDataResult;
  comparedRuns: RunRowType[];
  groupByColumns: string[];
  onCellClick?: (value: string, columnHeader: string) => void;
  onHideRun: (runUuid: string) => void;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
  highlightedText: string;
}

export const EvaluationArtifactCompareTable = ({
  resultList,
  comparedRuns,
  groupByColumns,
  onCellClick,
  onHideRun,
  onDatasetSelected,
  highlightedText = '',
}: EvaluationArtifactCompareTableProps) => {
  const [columns, setColumns] = useState<ColDef[]>([]);

  useEffect(() => {
    const cols: ColDef[] = [];

    groupByColumns.forEach((col) => {
      cols.push({
        width: EVALUATION_ARTIFACTS_TEXT_COLUMN_WIDTH,
        headerName: col,
        field: col,
        pinned: true,
        suppressMovable: true,
        cellRenderer: 'TextRendererCellRenderer',
        cellRendererParams: {
          highlightEnabled: true,
        },
        headerComponent: 'GroupHeaderCellRenderer',
        colId: col,
      });
    });

    for (const run of comparedRuns) {
      cols.push({
        width: EVALUATION_ARTIFACTS_TEXT_COLUMN_WIDTH,
        headerName: run.runName,
        colId: run.runUuid,
        field: run.runUuid,
        suppressMovable: true,
        cellRenderer: 'TextRendererCellRenderer',
        headerComponent: 'RunHeaderCellRenderer',
        headerComponentParams: { run, onHideRun, onDatasetSelected },
      });
    }

    setColumns(cols);
  }, [comparedRuns, groupByColumns, onHideRun, onDatasetSelected]);

  const { theme } = useDesignSystemTheme();

  const resultData = useMemo(() => {
    return resultList.map(({ cellValues, key, groupByCellValues }) => {
      const result: Record<string, string> & { key: string } = { key };
      Object.entries(groupByCellValues).forEach(([groupByKey, groupByValue]) => {
        result[groupByKey] = groupByValue;
      });
      Object.entries(cellValues).forEach(([runUuid, cellValue]) => {
        result[runUuid] = cellValue;
      });
      return result;
    });
  }, [resultList]);

  const handleCellClicked = ({ value, colDef, column }: CellClickedEvent) =>
    onCellClick?.(value, colDef.headerName || column.getId());

  return (
    <div css={{ height: '100%', overflow: 'hidden' }}>
      <MLFlowAgGridLoader
        css={createTableStyles(theme)}
        context={{ highlightedText }}
        rowHeight={EVALUATION_ARTIFACTS_TABLE_ROW_HEIGHT}
        headerHeight={getEvaluationArtifactsTableHeaderHeight()}
        getRowId={({ data }) => data.key}
        suppressHorizontalScroll={false}
        columnDefs={columns}
        onCellClicked={handleCellClicked}
        rowData={resultData}
        components={{
          TextRendererCellRenderer: EvaluationTextCellRenderer,
          GroupHeaderCellRenderer: EvaluationGroupByHeaderCellRenderer,
          RunHeaderCellRenderer: EvaluationRunHeaderCellRenderer,
        }}
      />
    </div>
  );
};

const createTableStyles = (theme: Theme) => ({
  '.ag-header-row.ag-header-row-column': {
    '& > div': {
      borderBottom: `1px solid ${theme.colors.borderDecorative}`,
    },
  },
  '.ag-row': {
    borderBottom: `1px solid ${theme.colors.borderDecorative}`,
  },
});

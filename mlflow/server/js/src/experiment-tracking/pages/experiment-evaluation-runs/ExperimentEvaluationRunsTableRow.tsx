import React, { useCallback, type CSSProperties } from 'react';
import {
  TableRow,
  TableCell,
  useDesignSystemTheme,
  Typography,
  Button,
  ChevronRightIcon,
  ChevronDownIcon,
  Tooltip,
  Tag,
} from '@databricks/design-system';
import type { EvalRunsTableColumnDef } from './ExperimentEvaluationRunsTable.constants';
import {
  EVAL_RUNS_TIGHT_PADDING_COLUMN_IDS,
  EvalRunsTableColumnId,
  EvalRunsTableKeyedColumnPrefix,
} from './ExperimentEvaluationRunsTable.constants';
import type { Row } from '@tanstack/react-table';
import { flexRender } from '@tanstack/react-table';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import type { RunGroupByGroupingValue } from '../../components/experiment-page/utils/experimentPage.row-types';
import { RunGroupingMode } from '../../components/experiment-page/utils/experimentPage.row-types';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';

type TracesViewTableRowProps = {
  row: Row<RunEntityOrGroupData>;
  isActive: boolean;
  // use for memoization updates to the checkbox
  isSelected: boolean;
  isExpanded: boolean;
  columns: EvalRunsTableColumnDef[];
  isHidden: boolean;
  isGrouped?: boolean;
  enableImprovedComparison?: boolean;
  /** Resizable-column styles keyed by column id; when present, cells size via the CSS-variable widths. */
  columnStyles?: Map<string, CSSProperties>;
  /** Makes the row span the summed column width on the resizable path (undefined on the fixed path). */
  rowWidthStyle?: CSSProperties;
  /** New UI: style of the empty trailing cell that takes the leftover row width. */
  fillerCellStyle?: CSSProperties;
  /** New UI: group rows get the secondary background and show per-metric averages. */
  showGroupSummaries?: boolean;
};

const GroupTag = ({ groupKey, groupValue }: { groupKey: string; groupValue: string }): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  return (
    <Tooltip content={groupKey + ': ' + groupValue} componentId="mlflow.eval-runs.group-tag">
      <Tag css={{ margin: 0 }} componentId="mlflow.eval-runs.group-tag">
        <Typography.Text
          bold
          css={{
            maxWidth: 100,
            marginRight: theme.spacing.xs,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {groupKey}:
        </Typography.Text>
        <Typography.Text css={{ maxWidth: 100, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {groupValue}
        </Typography.Text>
      </Tag>
    </Tooltip>
  );
};

const GroupLabel = ({ groupValues }: { groupValues: RunGroupByGroupingValue }): React.ReactElement => {
  const key = groupValues.groupByData;
  if (groupValues.mode === RunGroupingMode.Dataset) {
    return <GroupTag key={key} groupKey="Dataset" groupValue={String(groupValues.value)} />;
  }

  return <GroupTag key={key} groupKey={key} groupValue={String(groupValues.value)} />;
};

export const ExperimentEvaluationRunsTableRow = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    row,
    isActive,
    isGrouped,
    enableImprovedComparison,
    columns,
    columnStyles,
    rowWidthStyle,
    fillerCellStyle,
    showGroupSummaries,
  }: TracesViewTableRowProps) => {
    const { theme } = useDesignSystemTheme();
    const navigate = useNavigate();
    const getColumnStyles = (columnId: string) => columns.find((column) => column.id === columnId)?.meta?.styles;
    // Data cells align with the header; icon-only columns keep the tight default.
    const getCellStyle = (columnId: string) => {
      const baseStyle = columnStyles?.get(columnId);
      return columnStyles && !EVAL_RUNS_TIGHT_PADDING_COLUMN_IDS.has(columnId)
        ? { ...baseStyle, paddingLeft: theme.spacing.sm, paddingRight: theme.spacing.sm }
        : baseStyle;
    };

    // Row click navigation - only enabled when feature flag is on
    const handleRowClick = useCallback(() => {
      if (!enableImprovedComparison) {
        return;
      }
      if ('info' in row.original) {
        const { experimentId, runUuid } = row.original.info;
        navigate(Routes.getRunPageTabRoute(experimentId, runUuid, RunPageTabName.EVALUATIONS));
      }
    }, [enableImprovedComparison, navigate, row.original]);

    if ('groupValues' in row.original) {
      const groupLabel = (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.eval-runs.group-expand-button"
            size="small"
            css={{ flexShrink: 0 }}
            icon={row.getIsExpanded() ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={row.getToggleExpandedHandler()}
          />
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="Group:"
              description="Label for a group of runs in the evaluation runs table"
            />
          </Typography.Text>
          {row.original.groupValues.map((groupValue) => (
            <GroupLabel key={groupValue.groupByData} groupValues={groupValue} />
          ))}
        </div>
      );

      if (!showGroupSummaries) {
        return (
          <TableRow key={row.id} className="eval-runs-table-row">
            <TableCell>{groupLabel}</TableCell>
          </TableRow>
        );
      }

      return (
        <TableRow
          key={row.id}
          className="eval-runs-table-row"
          style={{ ...rowWidthStyle, backgroundColor: theme.colors.backgroundSecondary }}
        >
          {row.getVisibleCells().map((cell) => {
            const columnId = cell.column.id;
            const isLabelCell = columnId === EvalRunsTableColumnId.runName;
            const isMetricCell = columnId.startsWith(`${EvalRunsTableKeyedColumnPrefix.METRIC}.`);
            return (
              <TableCell
                key={cell.id}
                style={getCellStyle(columnId)}
                css={columnStyles ? undefined : getColumnStyles(columnId)}
              >
                {isLabelCell
                  ? groupLabel
                  : isMetricCell
                    ? flexRender(cell.column.columnDef.cell, cell.getContext())
                    : null}
              </TableCell>
            );
          })}
          {fillerCellStyle && <TableCell aria-hidden="true" style={fillerCellStyle} />}
        </TableRow>
      );
    }

    // Row click navigation and indent for grouped rows - only enabled when feature flag is on
    return (
      <TableRow
        key={row.id}
        className="eval-runs-table-row"
        onClick={enableImprovedComparison ? handleRowClick : undefined}
        style={rowWidthStyle}
        css={{
          cursor: enableImprovedComparison ? 'pointer' : undefined,
          marginLeft: enableImprovedComparison && isGrouped && row.depth > 0 ? theme.spacing.lg : undefined,
        }}
      >
        {row.getVisibleCells().map((cell) => (
          <TableCell
            key={cell.id}
            style={getCellStyle(cell.column.id)}
            css={[
              columnStyles ? undefined : (cell.column.columnDef as EvalRunsTableColumnDef).meta?.styles,
              {
                backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent',
              },
            ]}
          >
            {flexRender(cell.column.columnDef.cell, cell.getContext())}
          </TableCell>
        ))}
        {fillerCellStyle && (
          <TableCell
            aria-hidden="true"
            style={fillerCellStyle}
            css={{ backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent' }}
          />
        )}
      </TableRow>
    );
  },
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  (prev, next) => {
    return (
      prev.isActive === next.isActive &&
      prev.isSelected === next.isSelected &&
      prev.columns === next.columns &&
      prev.isExpanded === next.isExpanded &&
      prev.isHidden === next.isHidden &&
      prev.isGrouped === next.isGrouped &&
      prev.enableImprovedComparison === next.enableImprovedComparison &&
      prev.showGroupSummaries === next.showGroupSummaries &&
      prev.columnStyles === next.columnStyles &&
      // Group averages depend on the group's runs, which change as more pages load.
      (!next.showGroupSummaries || prev.row.original === next.row.original)
    );
  },
);

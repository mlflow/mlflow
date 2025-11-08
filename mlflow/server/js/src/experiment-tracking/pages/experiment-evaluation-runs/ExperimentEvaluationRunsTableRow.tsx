import React from 'react';
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
import type { Row } from '@tanstack/react-table';
import { flexRender } from '@tanstack/react-table';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import type { RunGroupByGroupingValue } from '../../components/experiment-page/utils/experimentPage.row-types';
import { RunGroupingMode } from '../../components/experiment-page/utils/experimentPage.row-types';
import { FormattedMessage } from 'react-intl';

type TracesViewTableRowProps = {
  row: Row<RunEntityOrGroupData>;
  isActive: boolean;
  // use for memoization updates to the checkbox
  isSelected: boolean;
  isExpanded: boolean;
  columns: EvalRunsTableColumnDef[];
  isHidden: boolean;
};

const GroupTag = ({ groupKey, groupValue }: { groupKey: string; groupValue: string }): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  return (
    <Tooltip content={groupKey + ': ' + groupValue} componentId={`mlflow.eval-runs.${groupKey}-group-tag`}>
      <Tag css={{ margin: 0 }} componentId={`mlflow.eval-runs.${groupKey}-group-tag`}>
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
  ({ row, isActive }: TracesViewTableRowProps) => {
    const { theme } = useDesignSystemTheme();

    if ('groupValues' in row.original) {
      return (
        <TableRow key={row.id} className="eval-runs-table-row">
          <TableCell>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Button
                componentId={`mlflow.eval-runs.${row.id}-group-expand-button`}
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
          </TableCell>
        </TableRow>
      );
    }

    return (
      <TableRow key={row.id} className="eval-runs-table-row">
        {row.getVisibleCells().map((cell) => (
          <TableCell
            key={cell.id}
            css={[
              (cell.column.columnDef as EvalRunsTableColumnDef).meta?.styles,
              {
                backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent',
              },
            ]}
          >
            {flexRender(cell.column.columnDef.cell, cell.getContext())}
          </TableCell>
        ))}
      </TableRow>
    );
  },
  (prev, next) => {
    return (
      prev.isActive === next.isActive &&
      prev.isSelected === next.isSelected &&
      prev.columns === next.columns &&
      prev.isExpanded === next.isExpanded &&
      prev.isHidden === next.isHidden
    );
  },
);

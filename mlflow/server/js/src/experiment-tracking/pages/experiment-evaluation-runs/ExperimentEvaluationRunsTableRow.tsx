import React, { useCallback } from 'react';
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
import { EvalRunsTableColumnId, EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';
import type { Row } from '@tanstack/react-table';
import { flexRender } from '@tanstack/react-table';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import type { RunGroupByGroupingValue } from '../../components/experiment-page/utils/experimentPage.row-types';
import { RunGroupingMode } from '../../components/experiment-page/utils/experimentPage.row-types';
import { FormattedMessage, useIntl } from 'react-intl';
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
  isBaseline?: boolean;
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
  const intl = useIntl();
  // Runs missing the grouping tag land here. They must be labelled rather than
  // shown as "null", because on an experiment where grouping was only just
  // introduced this bucket holds nearly everything.
  const value =
    groupValues.value === null || groupValues.value === undefined
      ? intl.formatMessage({
          defaultMessage: '(no value)',
          description: 'Group label for runs that do not carry the tag being grouped by',
        })
      : String(groupValues.value);

  if (groupValues.mode === RunGroupingMode.Dataset) {
    return <GroupTag key={key} groupKey="Dataset" groupValue={value} />;
  }

  return <GroupTag key={key} groupKey={key} groupValue={value} />;
};

export const ExperimentEvaluationRunsTableRow = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({ row, isActive, isGrouped, enableImprovedComparison, isBaseline }: TracesViewTableRowProps) => {
    const { theme } = useDesignSystemTheme();
    const navigate = useNavigate();

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
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="{count, plural, =1 {1 run} other {# runs}}"
              description="Count of runs contained in a group row of the evaluation runs table"
              values={{ count: row.original.subRuns.length }}
            />
          </Typography.Text>
        </div>
      );

      // Render every column, not one spanning cell: the group's per-metric means
      // have to sit directly under the metric they average, or they cannot be
      // compared down the column.
      return (
        <TableRow
          key={row.id}
          className="eval-runs-table-row"
          css={{ backgroundColor: theme.colors.backgroundPrimary }}
        >
          {row.getVisibleCells().map((cell) => {
            const columnId = cell.column.id;
            const isLabelCell = columnId === EvalRunsTableColumnId.runName;
            // Metric cells render the group's mean; everything left of the
            // metrics is chrome that would only repeat what the label says.
            const isMetricCell = columnId.startsWith(EvalRunsTableKeyedColumnPrefix.METRIC + '.');
            return (
              <TableCell key={cell.id} css={(cell.column.columnDef as EvalRunsTableColumnDef).meta?.styles}>
                {isLabelCell
                  ? groupLabel
                  : isMetricCell
                    ? flexRender(cell.column.columnDef.cell, cell.getContext())
                    : null}
              </TableCell>
            );
          })}
        </TableRow>
      );
    }

    // Row click navigation and indent for grouped rows - only enabled when feature flag is on
    return (
      <TableRow
        key={row.id}
        className="eval-runs-table-row"
        onClick={enableImprovedComparison ? handleRowClick : undefined}
        css={[
          {
            cursor: enableImprovedComparison ? 'pointer' : undefined,
            marginLeft: enableImprovedComparison && isGrouped && row.depth > 0 ? theme.spacing.lg : undefined,
          },
          // Three cues, all needed: the blue tint (never grey — grey already
          // means row selection, and a row can be both at once), a left accent
          // bar that survives when the tints are hard to tell apart, and a rule
          // separating the band from the sorted list beneath it.
          isBaseline && {
            backgroundColor: theme.colors.backgroundSecondary,
            boxShadow: `inset 3px 0 0 ${theme.colors.primary}`,
            borderBottom: `2px solid ${theme.colors.primary}`,
          },
        ]}
      >
        {row.getVisibleCells().map((cell) => (
          <TableCell
            key={cell.id}
            css={[
              (cell.column.columnDef as EvalRunsTableColumnDef).meta?.styles,
              {
                // Leave the cell transparent on the baseline row so the row's
                // own tint shows through instead of being painted over.
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
      prev.isBaseline === next.isBaseline
    );
  },
);

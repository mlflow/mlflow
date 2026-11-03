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
import { FormattedMessage, useIntl } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';
import {
  EVAL_RUN_AGENT_GROUP_BY_KEY,
  EVAL_RUN_SCORERS_GROUP_BY_KEY,
  formatShortDigest,
  parseScorersGroupValue,
} from '../eval-run-details/evalRunLineage';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

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
  /** Columns that a group row's label spans (from the run name to the first metric), and the span's style. */
  groupLabelSpan?: { columnIds: Set<string>; style: CSSProperties };
  /** Makes the row span the summed column width on the resizable path (undefined on the fixed path). */
  rowWidthStyle?: CSSProperties;
  /** New UI: style of the empty trailing cell that takes the leftover row width. */
  fillerCellStyle?: CSSProperties;
  /** New UI: group rows get the secondary background and show per-metric averages. */
  showGroupSummaries?: boolean;
  /** New UI: the experiment's baseline run, highlighted and pinned to the top. */
  isBaseline?: boolean;
};

const GROUP_TAG_TEXT_MAX_WIDTH = 280;
// The height of the small expand button, so that "Group:" lines up with the first row of tags.
const GROUP_ROW_LINE_HEIGHT = 24;

const GroupTag = ({
  groupKey,
  groupValue,
  fullGroupValue = groupValue,
}: {
  groupKey: string;
  groupValue: string;
  /** The untruncated value for the tooltip, when `groupValue` is a shortened form. */
  fullGroupValue?: string;
}): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', minHeight: GROUP_ROW_LINE_HEIGHT, maxWidth: '100%' }}>
      <Tooltip content={groupKey + ': ' + fullGroupValue} componentId="mlflow.eval-runs.group-tag">
        <Tag css={{ margin: 0 }} componentId="mlflow.eval-runs.group-tag">
          <Typography.Text
            css={{
              display: 'block',
              // A Tag doesn't shrink below its content, so cap the text to the tags container width
              // (`cqw`) minus the Tag's horizontal padding.
              maxWidth: `min(${GROUP_TAG_TEXT_MAX_WIDTH}px, calc(100cqw - ${theme.spacing.md}px))`,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            <Typography.Text bold>{groupKey}:</Typography.Text> {groupValue}
          </Typography.Text>
        </Tag>
      </Tooltip>
    </span>
  );
};

const GroupLabel = ({ groupValues }: { groupValues: RunGroupByGroupingValue }): React.ReactElement => {
  const intl = useIntl();
  const key = groupValues.groupByData;
  if (groupValues.mode === RunGroupingMode.Dataset) {
    return <GroupTag key={key} groupKey="Dataset" groupValue={String(groupValues.value)} />;
  }
  if (groupValues.mode === RunGroupingMode.Tag && key === EVAL_RUN_SCORERS_GROUP_BY_KEY) {
    const groupKey = intl.formatMessage({
      defaultMessage: 'Scorers',
      description: 'Label of an evaluation-runs group whose runs used the same scorers',
    });
    if (!groupValues.value) {
      return <GroupTag key={key} groupKey={groupKey} groupValue={String(groupValues.value)} />;
    }
    const { digest, hashed, total } = parseScorersGroupValue(String(groupValues.value));
    const formatDigest = (shownDigest: string) =>
      hashed === undefined
        ? shownDigest
        : intl.formatMessage(
            {
              defaultMessage: '{digest} (partial: {hashed} of {total})',
              description:
                'Value of an evaluation-runs "Scorers" group whose digest covers only the scorers that could be hashed',
            },
            { digest: shownDigest, hashed, total },
          );
    return (
      <GroupTag
        key={key}
        groupKey={groupKey}
        groupValue={formatDigest(formatShortDigest(digest))}
        fullGroupValue={formatDigest(digest)}
      />
    );
  }
  if (groupValues.mode === RunGroupingMode.Tag && key === EVAL_RUN_AGENT_GROUP_BY_KEY) {
    return (
      <GroupTag
        key={key}
        groupKey={intl.formatMessage({
          defaultMessage: 'Agent',
          description: 'Label of an evaluation-runs group whose runs evaluated the same agent',
        })}
        groupValue={String(groupValues.value)}
      />
    );
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
    groupLabelSpan,
    rowWidthStyle,
    fillerCellStyle,
    showGroupSummaries,
    isBaseline,
  }: TracesViewTableRowProps) => {
    const { theme } = useDesignSystemTheme();
    const rowBackgroundColor = isActive
      ? theme.colors.actionDefaultBackgroundHover
      : isBaseline
        ? theme.isDarkMode
          ? theme.colors.blue800
          : theme.colors.blue100
        : 'transparent';
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
        if (shouldEnableNewEvalRunsUi()) {
          navigate(Routes.getEvaluationRunDetailsRoute(experimentId, runUuid));
          return;
        }
        navigate(Routes.getRunPageTabRoute(experimentId, runUuid, RunPageTabName.EVALUATIONS));
      }
    }, [enableImprovedComparison, navigate, row.original]);

    if ('groupValues' in row.original) {
      const groupLabel = (
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.eval-runs.group-expand-button"
            size="small"
            css={{ flexShrink: 0 }}
            icon={row.getIsExpanded() ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={row.getToggleExpandedHandler()}
          />
          <div css={{ display: 'flex', alignItems: 'center', flexShrink: 0, minHeight: GROUP_ROW_LINE_HEIGHT }}>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Group:"
                description="Label for a group of runs in the evaluation runs table"
              />
            </Typography.Text>
          </div>
          {/* Tags stay on the "Group:" line and wrap under the first tag; the container query
              lets each tag truncate to the space that is left. */}
          <div
            css={{
              flex: '1 1 0',
              minWidth: 0,
              containerType: 'inline-size',
              display: 'flex',
              flexWrap: 'wrap',
              gap: theme.spacing.xs,
            }}
          >
            {row.original.groupValues.map((groupValue) => (
              <GroupLabel key={groupValue.groupByData} groupValues={groupValue} />
            ))}
          </div>
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
            if (!isLabelCell && groupLabelSpan?.columnIds.has(columnId)) {
              return null;
            }
            const isMetricCell = columnId.startsWith(`${EvalRunsTableKeyedColumnPrefix.METRIC}.`);
            return (
              <TableCell
                key={cell.id}
                aria-colspan={isLabelCell && groupLabelSpan ? groupLabelSpan.columnIds.size : undefined}
                style={
                  isLabelCell && groupLabelSpan
                    ? { ...groupLabelSpan.style, paddingLeft: theme.spacing.sm, paddingRight: theme.spacing.sm }
                    : getCellStyle(columnId)
                }
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
                backgroundColor: rowBackgroundColor,
              },
            ]}
          >
            {flexRender(cell.column.columnDef.cell, cell.getContext())}
          </TableCell>
        ))}
        {fillerCellStyle && (
          <TableCell aria-hidden="true" style={fillerCellStyle} css={{ backgroundColor: rowBackgroundColor }} />
        )}
      </TableRow>
    );
  },
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  (prev, next) => {
    return (
      prev.isActive === next.isActive &&
      prev.isBaseline === next.isBaseline &&
      prev.isSelected === next.isSelected &&
      prev.columns === next.columns &&
      prev.isExpanded === next.isExpanded &&
      prev.isHidden === next.isHidden &&
      prev.isGrouped === next.isGrouped &&
      prev.enableImprovedComparison === next.enableImprovedComparison &&
      prev.showGroupSummaries === next.showGroupSummaries &&
      prev.columnStyles === next.columnStyles &&
      prev.groupLabelSpan === next.groupLabelSpan &&
      // Group averages depend on the group's runs, which change as more pages load.
      (!next.showGroupSummaries || prev.row.original === next.row.original)
    );
  },
);

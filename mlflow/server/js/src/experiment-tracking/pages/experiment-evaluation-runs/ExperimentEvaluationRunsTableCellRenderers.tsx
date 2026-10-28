import {
  BeakerIcon,
  ChartLineIcon,
  DatabaseIcon,
  GavelIcon,
  ModelsIcon,
  TableIcon,
  TableClockIcon,
  Tag,
  Tooltip,
  TruncationTooltip,
  Typography,
  useDesignSystemTheme,
  Checkbox,
  ParagraphSkeleton,
  Button,
  NewWindowIcon,
  SortUnsortedIcon,
  VisibleIcon,
  VisibleOffIcon,
  SparkleIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  HoverCard,
} from '@databricks/design-system';
import type { CellContext, ColumnDef, HeaderContext } from '@tanstack/react-table';
import { DatasetSourceTypes, DEFAULT_INLINE_DATASET_NAME, RunEntity } from '../../types';
import { Link, useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetLoggedModelQuery } from '../../hooks/logged-models/useGetLoggedModelQuery';
import Routes from '../../routes';
import { useSaveExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { TimeAgo } from '@databricks/web-shared/browse';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { Fragment, useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import type { DatasetWithRunType } from '../../components/experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { isEvalRunsTableMeta } from './ExperimentEvaluationRunsTable.types';
import { getEvaluationDatasetId } from '../eval-run-details/datasetLineage';
import { EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';
import { formatEvalRunsNumericValue } from './ExperimentEvaluationRunsNumberFormat';
import { pinBaselineRows, toggleEvalRunRowSelection } from './evalRunsRowSelection';
import { getEvalRunsGroupMetricMean } from './EvalRunsGroup.utils';
import { useExperimentEvaluationRunsRowVisibility } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { RunPageTabName } from '../../constants';
import { shouldEnableImprovedEvalRunsComparison } from '../../../common/utils/FeatureUtils';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_TEST } from '../../constants';
import { DatasetLink } from '../experiment-evaluation-datasets/DatasetLink';
import { RunStatusIcon } from '../../components/RunStatusIcon';
import { useIntl } from '@databricks/i18n';

export const CheckboxCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row, table }) => {
  const { meta } = table.options;
  const intl = useIntl();
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const enableNewEvalRunsUi = isEvalRunsTableMeta(meta) && Boolean(meta.enableNewEvalRunsUi);
  const runLabel = row.original.info?.runName ?? row.original.info?.runUuid ?? '';

  return (
    <Checkbox
      componentId="mlflow.eval-runs.checkbox-cell"
      data-testid={`eval-runs-table-cell-checkbox-${row.id}`}
      aria-label={
        enableNewEvalRunsUi
          ? intl.formatMessage(
              {
                defaultMessage: 'Select run {runLabel}',
                description: 'Accessible label for a per-row selection checkbox in the evaluation runs table',
              },
              { runLabel },
            )
          : undefined
      }
      disabled={!row.getCanSelect()}
      isChecked={row.getIsSelected()}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={(_, event) => {
        const anchorRowIdRef = isEvalRunsTableMeta(meta) ? meta.selectionAnchorRowIdRef : undefined;
        if (!anchorRowIdRef) {
          row.toggleSelected();
          return;
        }
        const baselineRunUuid = isEvalRunsTableMeta(meta) ? meta.baselineRunUuid : undefined;
        toggleEvalRunRowSelection({
          table,
          row,
          selectRange: event.nativeEvent.shiftKey,
          anchorRowIdRef,
          orderRows: baselineRunUuid ? (rows) => pinBaselineRows(rows, baselineRunUuid) : undefined,
        });
      }}
      onClick={(e) => e.stopPropagation()}
    />
  );
};

export const RunNameCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({
  row,
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();
  const saveRunColor = useSaveExperimentRunColor();
  const getRunColor = useGetExperimentRunColor();
  const navigate = useNavigate();
  const intl = useIntl();

  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const runUuid = row.original.info.runUuid;
  const experimentId = row.original.info.experimentId;
  const evalMeta = isEvalRunsTableMeta(meta) ? meta : undefined;
  const enableNewEvalRunsUi = Boolean(evalMeta?.enableNewEvalRunsUi);
  const status = row.original.info.status;
  // Reserve a chevron-sized slot on non-expandable rows so they still indent by depth.
  const hasExpandableRows = Boolean(evalMeta?.hasExpandableRows);
  const reserveChevronSlot = enableNewEvalRunsUi && hasExpandableRows && !row.getCanExpand();
  const tags = row.original.data?.tags ?? [];

  // New UI renders run names as a real anchor so browser gestures (Cmd/Ctrl+click, middle-click,
  // "copy link") work.
  const runLinkTo = enableNewEvalRunsUi ? Routes.getEvaluationRunDetailsRoute(experimentId, runUuid) : undefined;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (runLinkTo) {
      navigate(runLinkTo);
      return;
    }
    // Otherwise follow old behavior - open the right-side panel
    evalMeta?.setSelectedRunUuid(runUuid);
  };

  const level = row.depth;
  const hasExpander = row.getCanExpand();
  const isExpanded = row.getIsExpanded();

  const handleExpanderClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    row.toggleExpanded();
  };

  const runNameLinkStyle = {
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    flexShrink: 1,
  } as const;

  const runNameLink = runLinkTo ? (
    <Link
      css={runNameLinkStyle}
      componentId="mlflow.eval-runs.run-name-cell"
      id="run-name-cell"
      to={runLinkTo}
      onClick={(e) => e.stopPropagation()}
    >
      {row.original.info.runName}
    </Link>
  ) : (
    <Typography.Link css={runNameLinkStyle} componentId="mlflow.eval-runs.run-name-cell" id="run-name-cell">
      {row.original.info.runName}
    </Typography.Link>
  );

  return (
    <div
      css={{
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        gap: enableNewEvalRunsUi ? theme.spacing.sm : theme.spacing.xs,
        paddingLeft: enableNewEvalRunsUi
          ? // Indent only the run-name column for nested rows so sibling columns stay aligned.
            level > 0
            ? theme.spacing.lg * level
            : undefined
          : level > 0
            ? (level + 1) * theme.spacing.lg
            : undefined,
      }}
      onClick={handleClick}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {hasExpander && (
        <Button
          componentId="mlflow.eval-runs.parent-child-expand-button"
          size="small"
          icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          aria-expanded={isExpanded}
          aria-label={
            isExpanded
              ? intl.formatMessage({
                  defaultMessage: 'Collapse child runs',
                  description: 'Aria label to collapse child runs',
                })
              : intl.formatMessage({
                  defaultMessage: 'Expand child runs',
                  description: 'Aria label to expand child runs',
                })
          }
          onClick={handleExpanderClick}
          type="link"
        />
      )}
      {reserveChevronSlot && (
        // Empty stand-in matching the expand button's footprint so leaf rows align with expandable
        // siblings and children sit one level in from their parent.
        <div aria-hidden css={{ width: theme.spacing.lg, flexShrink: 0 }} />
      )}
      {enableNewEvalRunsUi && status ? (
        <div css={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
          <RunStatusIcon status={status} />
        </div>
      ) : null}
      <RunColorPill
        color={getRunColor(runUuid)}
        onChangeColor={(colorValue) => saveRunColor({ runUuid, colorValue })}
      />
      {enableNewEvalRunsUi ? (
        <TruncationTooltip componentId="mlflow.eval-runs.run-name-cell.truncation-tooltip">
          {runNameLink}
        </TruncationTooltip>
      ) : (
        runNameLink
      )}
      {enableNewEvalRunsUi && evalMeta?.baselineRunUuid === runUuid ? (
        <Tag componentId="mlflow.eval-runs.baseline-tag">
          <Typography.Text size="sm" bold>
            <FormattedMessage defaultMessage="BASELINE" description="Label for the baseline evaluation run" />
          </Typography.Text>
        </Tag>
      ) : null}
      {!shouldEnableImprovedEvalRunsComparison() && (
        <div
          css={{
            display: 'none',
            flexShrink: 0,
            '.eval-runs-table-row:hover &': { display: 'inline' },
            svg: {
              width: theme.typography.fontSizeMd,
              height: theme.typography.fontSizeMd,
            },
          }}
        >
          <Link
            componentId="mlflow.experiment_tracking.evaluation_runs.run_link"
            target="_blank"
            rel="noreferrer"
            to={Routes.getRunPageTabRoute(row.original.info.experimentId, runUuid, RunPageTabName.EVALUATIONS)}
            onClick={(e) => e.stopPropagation()}
          >
            <Tooltip
              content={
                <FormattedMessage
                  defaultMessage="Go to the run"
                  description="Tooltip for the run name cell in the evaluation runs table, opening the run page in a new tab"
                />
              }
              componentId="mlflow.eval-runs.run-name-cell.tooltip"
            >
              <Button
                type="link"
                target="_blank"
                icon={<NewWindowIcon />}
                size="small"
                componentId="mlflow.eval-runs.run-name-cell.open-run-page"
              />
            </Tooltip>
          </Link>
        </div>
      )}
    </div>
  );
};

/** Hover card on the dataset pill with the new evaluation runs UI: name, digest and source type, plus a link to
 * the evaluation dataset when the input was logged from one. */
const DatasetHoverCardContent = ({
  experimentId,
  label,
  isUnnamed,
  dataset,
}: {
  experimentId?: string;
  label: React.ReactNode;
  isUnnamed: boolean;
  dataset: { digest?: string | null; source?: string | null; sourceType?: string | null };
}) => {
  const { theme } = useDesignSystemTheme();
  const datasetId = getEvaluationDatasetId(dataset.source);
  const metadata = [
    {
      key: 'digest',
      label: <FormattedMessage defaultMessage="Digest" description="Label for the digest in a dataset hover card" />,
      value: dataset.digest || '-',
    },
    {
      key: 'source_type',
      label: (
        <FormattedMessage
          defaultMessage="Source type"
          description="Label for the source type in a dataset hover card"
        />
      ),
      value: dataset.sourceType || '-',
    },
  ];

  return (
    <div css={{ width: 280 }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
        {isUnnamed ? (
          <TableClockIcon css={{ flexShrink: 0, color: theme.colors.textSecondary }} />
        ) : (
          <DatabaseIcon css={{ flexShrink: 0, color: theme.colors.textSecondary }} />
        )}
        <Typography.Text bold css={{ overflowWrap: 'anywhere' }}>
          {label}
        </Typography.Text>
      </div>
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'max-content minmax(0, 1fr)',
          columnGap: theme.spacing.lg,
          rowGap: theme.spacing.xs,
        }}
      >
        {metadata.map(({ key, label: metadataLabel, value }) => (
          <Fragment key={key}>
            <Typography.Text color="secondary">{metadataLabel}</Typography.Text>
            <Typography.Text css={{ overflowWrap: 'anywhere' }}>{value}</Typography.Text>
          </Fragment>
        ))}
      </div>
      {datasetId && experimentId && (
        <Link
          componentId="mlflow.eval-runs.dataset-cell.hover-card.view-dataset"
          to={Routes.getExperimentPageDatasetDetailRoute(experimentId, datasetId)}
          target="_blank"
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            marginTop: theme.spacing.sm,
            paddingTop: theme.spacing.sm,
            borderTop: `1px solid ${theme.colors.border}`,
          }}
        >
          <FormattedMessage
            defaultMessage="View dataset"
            description="Link in the dataset hover card that opens the evaluation dataset"
          />
          <NewWindowIcon />
        </Link>
      )}
    </div>
  );
};

export const DatasetCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({
  row,
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();
  const evalMeta = isEvalRunsTableMeta(meta) ? meta : undefined;
  const enableNewEvalRunsUi = Boolean(evalMeta?.enableNewEvalRunsUi);

  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const run = row.original;
  const datasets = run.inputs?.datasetInputs ?? [];
  const displayedDataset = datasets[0]?.dataset ?? null;

  if (!displayedDataset) {
    return <div>-</div>;
  }

  // Data passed in at eval time (inline df/list → CodeDatasetSource, or a local file → 'local') is
  // logged with MLflow's generic default name, so there's nothing meaningful to show. Treat any such
  // dataset as "unnamed" and surface its source type instead; named datasets show their real name.
  const isUnnamedDataset =
    enableNewEvalRunsUi && (!displayedDataset.name || displayedDataset.name === DEFAULT_INLINE_DATASET_NAME);
  const datasetLabel = isUnnamedDataset ? (
    <FormattedMessage
      defaultMessage="Unnamed dataset"
      description="Eval runs table > dataset column > label for a dataset logged without a meaningful name (e.g. inline code/local data)"
    />
  ) : (
    displayedDataset.name
  );

  const openDatasetDrawer = (e: React.MouseEvent) => {
    e.stopPropagation();
    evalMeta?.setSelectedDatasetWithRun({
      // The drawer renders the dataset context tag without a flag guard, so pass the real tags only
      // under the new UI; the legacy path keeps passing the dataset alone.
      datasetWithTags: (enableNewEvalRunsUi
        ? { dataset: displayedDataset, tags: datasets[0]?.tags ?? [] }
        : { dataset: displayedDataset }) as DatasetWithRunType['datasetWithTags'],
      runData: {
        experimentId: run.info?.experimentId,
        runUuid: run.info?.runUuid ?? '',
        runName: run.info?.runName,
        datasets: datasets,
      },
    });
    evalMeta?.setIsDrawerOpen(true);
  };

  const baseTagContent = (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        maxWidth: '100%',
        color: theme.colors.textPrimary,
      }}
    >
      {isUnnamedDataset ? (
        <TableClockIcon css={{ '& > svg': { width: 12, height: 12 } }} />
      ) : enableNewEvalRunsUi ? (
        <DatabaseIcon css={{ '& > svg': { width: 12, height: 12 } }} />
      ) : (
        <TableIcon css={{ '& > svg': { width: 12, height: 12 } }} />
      )}
      <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
        {datasetLabel}
      </Typography.Text>
    </div>
  );
  const tagContent = (
    // The link wrapper must not grow past the pill, or long dataset names overflow instead of truncating.
    <DatasetLink dataset={displayedDataset} {...(enableNewEvalRunsUi ? { css: { maxWidth: '100%' } } : {})}>
      {baseTagContent}
    </DatasetLink>
  );

  const datasetTag = (
    <Tag
      componentId="mlflow.eval-runs.dataset-cell"
      onClick={openDatasetDrawer}
      id="dataset-cell"
      css={{
        maxWidth: '100%',
        marginRight: 0,
        ...(enableNewEvalRunsUi
          ? {
              backgroundColor: theme.colors.backgroundSecondary,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
            }
          : {}),
      }}
    >
      {tagContent}
    </Tag>
  );

  return (
    <div>
      {enableNewEvalRunsUi ? (
        // Always surface the source type and digest alongside the name/label: for unnamed datasets they're the
        // only identifying info, and for named ones they're still useful context.
        <HoverCard
          trigger={datasetTag}
          content={
            <DatasetHoverCardContent
              experimentId={run.info?.experimentId}
              label={datasetLabel}
              isUnnamed={isUnnamedDataset}
              dataset={displayedDataset}
            />
          }
        />
      ) : (
        <Tooltip componentId="mlflow.eval-runs.dataset-cell-tooltip" content={displayedDataset.name}>
          {datasetTag}
        </Tooltip>
      )}
    </div>
  );
};

export const DatasetVersionCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row }) => {
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  // Version == the input dataset's digest; eval runs have at most one dataset input.
  const digest = row.original.inputs?.datasetInputs?.[0]?.dataset?.digest;
  if (!digest) {
    return <div>-</div>;
  }

  return (
    <Tooltip componentId="mlflow.eval-runs.dataset-version-cell-tooltip" content={digest}>
      <Typography.Text
        color="secondary"
        css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block' }}
      >
        {digest}
      </Typography.Text>
    </Tooltip>
  );
};

export const ModelVersionCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row }) => {
  const modelId = 'inputs' in row.original ? row.original.inputs?.modelInputs?.[0]?.modelId : undefined;
  const { theme } = useDesignSystemTheme();
  const { data, isLoading } = useGetLoggedModelQuery({ loggedModelId: modelId, enabled: Boolean(modelId) });

  if (!modelId || 'subRuns' in row.original) {
    return <div>-</div>;
  }

  const displayValue = data?.info?.name ?? modelId;

  return isLoading ? (
    <ParagraphSkeleton />
  ) : (
    <Tooltip componentId="mlflow.eval-runs.model-version-cell-tooltip" content={displayValue}>
      <Tag
        componentId="mlflow.eval-runs.model-version-cell"
        id="model-version-cell"
        css={{ maxWidth: '100%', marginRight: 0, cursor: 'pointer' }}
      >
        <Link
          componentId="mlflow.experiment_tracking.evaluation_runs.model_version_link"
          to={Routes.getExperimentLoggedModelDetailsPageRoute(row.original.info.experimentId, modelId)}
          target="_blank"
          css={{ maxWidth: '100%' }}
        >
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              maxWidth: '100%',
            }}
          >
            <ModelsIcon css={{ '& > svg': { width: 12, height: 12, color: theme.colors.textPrimary } }} />
            <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
              {displayValue}
            </Typography.Text>
          </div>
        </Link>
      </Tag>
    </Tooltip>
  );
};

export const KeyedValueCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({
  column,
  getValue,
  row,
  table,
}: CellContext<RunEntityOrGroupData, unknown>) => {
  const { theme } = useDesignSystemTheme();
  const tableMeta = table.options.meta;
  const enableNewEvalRunsUi = isEvalRunsTableMeta(tableMeta) && Boolean(tableMeta.enableNewEvalRunsUi);

  if (!enableNewEvalRunsUi) {
    const value = getValue<string>();
    return <span title={value}>{value ?? '-'}</span>;
  }

  const value = getValue<string | number | undefined>();
  const parsedColumn = parseEvalRunsTableKeyedColumnKey(column.id);
  const metricKey = parsedColumn?.columnType === EvalRunsTableKeyedColumnPrefix.METRIC ? parsedColumn.key : undefined;

  if ('subRuns' in row.original && metricKey) {
    const average = getEvalRunsGroupMetricMean(row.original.subRuns ?? [], metricKey);
    if (average === undefined) {
      return <span>-</span>;
    }
    const displayValue = formatEvalRunsNumericValue(average);
    return (
      <Tooltip componentId="mlflow.eval-runs.group-metric-average.tooltip" content={String(average)}>
        <span css={{ display: 'inline-flex', alignItems: 'center' }}>
          <FormattedMessage
            defaultMessage="<label>avg</label> <value>{average}</value>"
            description="Average metric value of a group of evaluation runs"
            values={{
              average: displayValue,
              label: (chunks) => (
                <Typography.Text
                  size="sm"
                  color="secondary"
                  css={{ marginRight: theme.spacing.xs, fontSize: theme.typography.fontSizeSm - 2 }}
                >
                  {chunks}
                </Typography.Text>
              ),
              value: (chunks) => <Typography.Text bold>{chunks}</Typography.Text>,
            }}
          />
        </span>
      </Tooltip>
    );
  }

  const displayValue = metricKey !== undefined && typeof value === 'number' ? formatEvalRunsNumericValue(value) : value;

  const valueText = (
    <span css={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
      {displayValue ?? '-'}
    </span>
  );
  // The cell rounds metrics, so a rounded value always gets a tooltip with the full value.
  if (value !== undefined && String(displayValue) !== String(value)) {
    return (
      <Tooltip componentId="mlflow.eval-runs.keyed-value-cell.tooltip" content={String(value)}>
        {valueText}
      </Tooltip>
    );
  }
  return (
    <TruncationTooltip componentId="mlflow.eval-runs.keyed-value-cell.truncation-tooltip">
      {valueText}
    </TruncationTooltip>
  );
};

export const SortableHeaderCell = ({
  column,
  table,
  title,
}: HeaderContext<RunEntityOrGroupData, unknown> & { title?: React.ReactElement }) => {
  const { theme } = useDesignSystemTheme();
  // Sorting is driven from the header menu, so suppress the built-in hover sort icon and pointer cursor.
  const meta = table.options.meta;
  const hideSortIcon = Boolean(isEvalRunsTableMeta(meta) && meta.enableNewEvalRunsUi);

  const parsedColumn = useMemo(() => parseEvalRunsTableKeyedColumnKey(column.id), [column.id]);
  const displayedKey = parsedColumn?.key ?? column.id;
  const isMetric = parsedColumn?.columnType === EvalRunsTableKeyedColumnPrefix.METRIC;

  return (
    <div
      css={{
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        ...(hideSortIcon ? {} : { ':hover': { cursor: 'pointer', '& > div': { display: 'inline' } } }),
      }}
    >
      {isMetric && hideSortIcon && (
        <Tooltip
          componentId="mlflow.eval-runs.metric-header-icon.tooltip"
          content={
            <FormattedMessage
              defaultMessage="Metric"
              description="Tooltip for the metric icon in evaluation runs table headers"
            />
          }
        >
          <GavelIcon aria-hidden="true" css={{ flexShrink: 0 }} />
        </Tooltip>
      )}
      <Tooltip
        componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_evaluation_runs_experimentevaluationrunstablecellrenderers_284"
        content={displayedKey}
      >
        <span css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
          <Typography.Text bold>{title ?? displayedKey}</Typography.Text>
        </span>
      </Tooltip>
      {!hideSortIcon && !column.getIsSorted() && (
        <div
          css={{
            display: 'none',
            flexShrink: 0,
          }}
        >
          <SortUnsortedIcon />
        </div>
      )}
    </div>
  );
};

export const CreatedAtCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row }) => {
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const createdAt = row.original.info.startTime;
  if (!createdAt) {
    return <div>-</div>;
  }
  return <TimeAgo date={new Date(Number(createdAt))} />;
};

export const VisiblityCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row, table }) => {
  const { isRowHidden, toggleRowVisibility } = useExperimentEvaluationRunsRowVisibility();
  // TODO: allow toggling visibility for a whole run group
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }
  const runUuid = row.original.info.runUuid;
  // The baseline is always shown, so its visibility can't be toggled.
  if (isEvalRunsTableMeta(table.options.meta) && table.options.meta.baselineRunUuid === runUuid) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', height: '100%' }}>
        <VisibleIcon />
      </div>
    );
  }

  // Use pre-calculated flat run index from table meta
  // This index excludes group rows and provides a global position across all parents
  const flatRunIndexMap = (table.options.meta as any)?.flatRunIndexMap;
  const rowIndex = flatRunIndexMap?.[runUuid] ?? row.index;

  const runStatus = row.original.info.status;
  const Icon = isRowHidden(runUuid, rowIndex, runStatus) ? VisibleOffIcon : VisibleIcon;

  return (
    <div css={{ display: 'flex', alignItems: 'center', height: '100%' }}>
      <Icon
        onClick={(e: React.MouseEvent) => {
          e.stopPropagation();
          toggleRowVisibility(runUuid, rowIndex, runStatus);
        }}
        css={{ cursor: 'pointer' }}
      />
    </div>
  );
};

export const StatusCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row }) => {
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const status = row.original.info.status;
  if (!status) {
    return <div>-</div>;
  }

  return <RunStatusIcon status={status} />;
};

/**
 * Renders the run's Type pill, keyed off the `mlflow.runType` tag:
 * - a purple "Test" pill with a beaker icon for `@mlflow.test` pytest runs,
 * - an indigo "Issue detection" pill with a sparkle icon for issue-detection runs,
 * - a turquoise "Eval" pill with a chart-line icon for everything else.
 * Lets users tell the run kinds apart at a glance.
 */
export const TypeCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row }) => {
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }
  const tags = row.original.data?.tags ?? [];
  const runType = tags.find((tag) => tag.key === MLFLOW_RUN_TYPE_TAG)?.value;
  const pillCss = { display: 'inline-flex', alignItems: 'center', gap: 4, margin: 0 } as const;

  switch (runType) {
    case MLFLOW_RUN_TYPE_VALUE_TEST:
      return (
        <Tag componentId="mlflow.eval-runs.type-cell.test" color="purple" css={pillCss}>
          <BeakerIcon />
          <FormattedMessage
            defaultMessage="Test"
            description="Type pill text for a regression-test run in the evaluation runs table"
          />
        </Tag>
      );
    default:
      return (
        <Tag componentId="mlflow.eval-runs.type-cell.eval" color="turquoise" css={pillCss}>
          <ChartLineIcon />
          <FormattedMessage
            defaultMessage="Eval"
            description="Type pill text for a regular evaluation run in the evaluation runs table"
          />
        </Tag>
      );
  }
};

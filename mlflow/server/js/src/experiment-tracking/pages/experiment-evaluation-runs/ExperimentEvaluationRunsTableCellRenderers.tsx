import {
  BeakerIcon,
  ChartLineIcon,
  ModelsIcon,
  TableIcon,
  Tag,
  Tooltip,
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
} from '@databricks/design-system';
import type { ColumnDef, HeaderContext } from '@tanstack/react-table';
import {
  formatEvalRunsMetric,
  getEvalRunsDelta,
  getRunMetricValue,
  type EvalRunsDelta,
} from './EvalRunsBaseline.utils';
import type { RunEntity } from '../../types';
import { DatasetSourceTypes } from '../../types';
import { Link, useNavigate, useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetLoggedModelQuery } from '../../hooks/logged-models/useGetLoggedModelQuery';
import Routes from '../../routes';
import { getPreservedQueryString } from '../experiment-page-tabs/side-nav/utils';
import { useSaveExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { TimeAgo } from '@databricks/web-shared/browse';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { useExperimentEvaluationRunsRowVisibility } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { RunPageTabName } from '../../constants';
import {
  shouldEnableImprovedEvalRunsComparison,
  shouldShowEvalRunsIssuesPanel,
} from '../../../common/utils/FeatureUtils';
import {
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
  MLFLOW_RUN_TYPE_VALUE_TEST,
} from '../../constants';
import { DatasetLink } from '../experiment-evaluation-datasets/DatasetLink';
import { RunStatusIcon } from '../../components/RunStatusIcon';

export const CheckboxCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({
  row,
  table: {
    options: { meta },
  },
}) => {
  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  return (
    <Checkbox
      componentId="mlflow.eval-runs.checkbox-cell"
      data-testid={`eval-runs-table-cell-checkbox-${row.id}`}
      disabled={!row.getCanSelect()}
      isChecked={row.getIsSelected()}
      wrapperStyle={{ padding: 0, margin: 0 }}
      onChange={() => row.toggleSelected()}
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
  const [searchParams] = useSearchParams();

  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const runUuid = row.original.info.runUuid;
  const experimentId = row.original.info.experimentId;
  const tags = row.original.data?.tags ?? [];
  const isIssueDetectionRun = tags.some(
    (tag) => tag.key === MLFLOW_RUN_TYPE_TAG && tag.value === MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
  );
  const showIssuesPanelFlag = shouldShowEvalRunsIssuesPanel();

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    // When flag is ON and clicking on an issue detection run, navigate to the issue detection run details page
    if (isIssueDetectionRun && showIssuesPanelFlag) {
      const route = Routes.getIssueDetectionRunDetailsRoute(experimentId, runUuid);
      const timeRangeSearch = getPreservedQueryString(searchParams.toString());
      navigate(timeRangeSearch ? `${route}${timeRangeSearch}` : route);
      return;
    }
    // Otherwise follow old behavior - open the right-side panel
    (meta as any).setSelectedRunUuid?.(runUuid);
  };

  return (
    <div
      css={{ overflow: 'hidden', display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
      onClick={handleClick}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {isIssueDetectionRun && showIssuesPanelFlag ? (
        <Tooltip
          content={
            <FormattedMessage
              defaultMessage="Issue detection run"
              description="Tooltip for the AI icon indicating this is an issue detection run"
            />
          }
          componentId="mlflow.eval-runs.issue-detection-run-icon-tooltip"
        >
          <SparkleIcon color="ai" css={{ flexShrink: 0, fontSize: 14, marginLeft: -1, marginRight: -1 }} />
        </Tooltip>
      ) : (
        <RunColorPill
          color={getRunColor(runUuid)}
          onChangeColor={(colorValue) => saveRunColor({ runUuid, colorValue })}
        />
      )}
      <Typography.Link
        css={{ textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden', flexShrink: 1 }}
        componentId="mlflow.eval-runs.run-name-cell"
        id="run-name-cell"
        // Long names still ellipsize at narrow widths; the native tooltip is the
        // only way to recover the full name without opening the run.
        title={row.original.info.runName}
      >
        {row.original.info.runName}
      </Typography.Link>
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

export const DatasetCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({
  row,
  table: {
    options: { meta },
  },
}) => {
  const { theme } = useDesignSystemTheme();

  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const run = row.original;
  const datasets = run.inputs?.datasetInputs ?? [];
  const displayedDataset = datasets[0]?.dataset ?? null;

  if (!displayedDataset) {
    return <div>-</div>;
  }

  const openDatasetDrawer = (e: React.MouseEvent) => {
    e.stopPropagation();
    (meta as any).setSelectedDatasetWithRun({
      datasetWithTags: { dataset: displayedDataset },
      runData: {
        experimentId: run.info?.experimentId,
        runUuid: run.info?.runUuid ?? '',
        runName: run.info?.runName,
        datasets: datasets,
      },
    });
    (meta as any).setIsDrawerOpen(true);
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
      <TableIcon css={{ '& > svg': { width: 12, height: 12 } }} />
      <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
        {displayedDataset.name}
      </Typography.Text>
    </div>
  );
  const tagContent = <DatasetLink dataset={displayedDataset}>{baseTagContent}</DatasetLink>;

  return (
    <div>
      <Tooltip componentId="mlflow.eval-runs.dataset-cell-tooltip" content={displayedDataset.name}>
        <Tag
          componentId="mlflow.eval-runs.dataset-cell"
          onClick={openDatasetDrawer}
          id="dataset-cell"
          css={{ maxWidth: '100%', marginRight: 0 }}
        >
          {tagContent}
        </Tag>
      </Tooltip>
    </div>
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

export const KeyedValueCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ getValue }) => {
  const value = getValue<string>();
  return <span title={value}>{value ?? '-'}</span>;
};

/**
 * Renders a ▲/▼ next to a metric value. The glyph carries direction on its own
 * so the signal survives for colour-blind users and in screenshots, and colour
 * reinforces whether the move was an improvement — which is not the same as
 * whether the number went up (see `isLowerBetterMetric`).
 */
const DeltaIndicator = ({
  delta,
  metricKey,
  isAverage = false,
}: {
  delta: EvalRunsDelta;
  metricKey: string;
  /** Group rows compare an average, so the tooltip must not claim a single run's value. */
  isAverage?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // A move under the noise floor is reported as unchanged rather than as a
  // coloured arrow, so run-to-run jitter doesn't read as a real improvement.
  if (delta.direction === 'neutral') {
    return (
      <Tooltip
        componentId="mlflow.eval-runs.delta.unchanged-tooltip"
        content={
          isAverage
            ? intl.formatMessage(
                {
                  defaultMessage: 'Group average unchanged vs baseline ({metricKey})',
                  description:
                    'Tooltip on a group row whose average delta against the baseline is below the noise floor',
                },
                { metricKey },
              )
            : intl.formatMessage(
                {
                  defaultMessage: 'Unchanged vs baseline ({metricKey})',
                  description: 'Tooltip on a metric whose delta against the baseline is below the noise floor',
                },
                { metricKey },
              )
        }
      >
        <span css={{ color: theme.colors.textSecondary, marginLeft: theme.spacing.xs }}>–</span>
      </Tooltip>
    );
  }

  const isBetter = delta.direction === 'better';
  const magnitude = formatEvalRunsMetric(Math.abs(delta.value));
  const signed = `${delta.value > 0 ? '+' : '−'}${magnitude}`;
  const verdict = isBetter
    ? intl.formatMessage({
        defaultMessage: 'better',
        description: 'Verdict when a metric improved relative to the baseline run',
      })
    : intl.formatMessage({
        defaultMessage: 'worse',
        description: 'Verdict when a metric regressed relative to the baseline run',
      });

  // Each formatMessage call needs its descriptor written inline: the formatjs
  // build step extracts the id statically, so a descriptor chosen by a ternary
  // reaches runtime without one and throws.
  const tooltipContent = isAverage
    ? intl.formatMessage(
        {
          defaultMessage: 'Group average {signed} vs baseline ({verdict})',
          description: "Tooltip showing the signed delta of a group's average metric against the baseline run",
        },
        { signed, verdict },
      )
    : intl.formatMessage(
        {
          defaultMessage: '{signed} vs baseline ({verdict})',
          description: 'Tooltip showing the signed delta of a metric against the baseline run',
        },
        { signed, verdict },
      );

  return (
    <Tooltip componentId="mlflow.eval-runs.delta.tooltip" content={tooltipContent}>
      {/* The arrow carries the direction, the number the magnitude. The arrow
          alone cannot separate +0.001 from +0.4, so the size has to be readable
          without hovering every cell. */}
      <span
        aria-label={signed}
        css={{
          display: 'inline-flex',
          alignItems: 'baseline',
          gap: 2,
          marginLeft: theme.spacing.xs,
          fontSize: theme.typography.fontSizeSm,
          // Tabular figures so the deltas form a straight column instead of
          // jittering with each digit's width.
          fontVariantNumeric: 'tabular-nums',
          color: isBetter ? theme.colors.textValidationSuccess : theme.colors.textValidationDanger,
        }}
      >
        <span aria-hidden>{delta.value > 0 ? '▲' : '▼'}</span>
        {/* The arrow already states the sign; repeating +/- here would say it
            twice in the same breath. */}
        <span aria-hidden>{magnitude}</span>
      </span>
    </Tooltip>
  );
};

/**
 * Metric cell for the baseline-aware table. Group rows show the mean of their
 * subruns, which is bounded to the group and so cannot blend populations the
 * way a whole-column average would across the 50-per-page infinite scroll.
 */
export const MetricValueCell: ColumnDef<RunEntityOrGroupData>['cell'] = ({ row, column, table }) => {
  const { theme } = useDesignSystemTheme();
  const meta = table.options.meta as any;
  const metricKey = parseEvalRunsTableKeyedColumnKey(column.id)?.key ?? column.id;
  const baselineRun: RunEntity | undefined = meta?.baselineRun;
  const baselineValue = getRunMetricValue(baselineRun, metricKey);

  if ('subRuns' in row.original) {
    const mean = meta?.getGroupMetricMean?.(row.original.subRuns, metricKey);
    if (!Number.isFinite(mean)) {
      return <div>-</div>;
    }
    // The group holding the baseline compares against itself, so its own run is
    // excluded from the average before the delta is taken. Left in, a group of
    // one would always read as unchanged and a larger group would be pulled
    // toward the very number it is measured against.
    const comparableRuns = baselineRun
      ? row.original.subRuns.filter((subRun) => subRun.info.runUuid !== baselineRun.info.runUuid)
      : row.original.subRuns;
    const comparableMean =
      comparableRuns.length === row.original.subRuns.length
        ? mean
        : meta?.getGroupMetricMean?.(comparableRuns, metricKey);
    const groupDelta = getEvalRunsDelta(metricKey, comparableMean, baselineValue);

    return (
      <span css={{ display: 'inline-flex', alignItems: 'baseline', gap: theme.spacing.xs }}>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="avg"
            description="Prefix marking a group row value as the average across the runs in that group"
          />
        </Typography.Text>
        <Typography.Text bold>{formatEvalRunsMetric(mean)}</Typography.Text>
        {groupDelta && <DeltaIndicator delta={groupDelta} metricKey={metricKey} isAverage />}
      </span>
    );
  }

  const value = getRunMetricValue(row.original, metricKey);
  if (!Number.isFinite(value)) {
    return <div>-</div>;
  }

  const isBaselineRow = baselineRun?.info?.runUuid === row.original.info.runUuid;
  const delta = isBaselineRow ? undefined : getEvalRunsDelta(metricKey, value, baselineValue);

  return (
    <span css={{ display: 'inline-flex', alignItems: 'center' }}>
      {formatEvalRunsMetric(value as number)}
      {delta && <DeltaIndicator delta={delta} metricKey={metricKey} />}
    </span>
  );
};

export const SortableHeaderCell = ({
  column,
  title,
}: HeaderContext<RunEntityOrGroupData, unknown> & { title?: React.ReactElement }) => {
  const { theme } = useDesignSystemTheme();

  const displayedKey = useMemo(() => parseEvalRunsTableKeyedColumnKey(column.id)?.key ?? column.id, [column.id]);

  return (
    <div
      css={{
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        ':hover': { cursor: 'pointer', '& > div': { display: 'inline' } },
      }}
    >
      <Tooltip
        componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_evaluation_runs_experimentevaluationrunstablecellrenderers_284"
        content={displayedKey}
      >
        <span css={{ overflow: 'hidden', textOverflow: 'ellipsis', textWrap: 'nowrap' }}>
          <Typography.Text bold>{title ?? displayedKey}</Typography.Text>
        </span>
      </Tooltip>
      {!column.getIsSorted() && (
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
  const rowIndex = row.index;
  const runStatus = row.original.info.status;
  const Icon = isRowHidden(runUuid, rowIndex, runStatus) ? VisibleOffIcon : VisibleIcon;

  return (
    <div css={{ display: 'flex', alignItems: 'center', height: '100%' }}>
      <Icon
        onClick={(e: React.MouseEvent) => {
          e.stopPropagation();
          toggleRowVisibility(runUuid);
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
    case MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION:
      return (
        <Tag componentId="mlflow.eval-runs.type-cell.issue-detection" color="indigo" css={pillCss}>
          <SparkleIcon />
          <FormattedMessage
            defaultMessage="Issue detection"
            description="Type pill text for an issue-detection run in the evaluation runs table"
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

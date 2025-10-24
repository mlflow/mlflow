import {
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
} from '@databricks/design-system';
import type { ColumnDef, HeaderContext } from '@tanstack/react-table';
import { DatasetSourceTypes, RunEntity } from '../../types';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetLoggedModelQuery } from '../../hooks/logged-models/useGetLoggedModelQuery';
import Routes from '../../routes';
import { FormattedMessage } from 'react-intl';
import { RunPageTabName } from '../../constants';
import { useSaveExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { TimeAgo } from '@databricks/web-shared/browse';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { useMemo } from 'react';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { useExperimentEvaluationRunsRowVisibility } from './hooks/useExperimentEvaluationRunsRowVisibility';

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

  if ('subRuns' in row.original) {
    return <div>-</div>;
  }

  const runUuid = row.original.info.runUuid;

  return (
    <div
      css={{ overflow: 'hidden', display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
      onClick={() => {
        (meta as any).setSelectedRunUuid?.(runUuid);
      }}
    >
      <RunColorPill
        color={getRunColor(runUuid)}
        onChangeColor={(colorValue) => saveRunColor({ runUuid, colorValue })}
      />
      <Typography.Link
        css={{ textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden', flexShrink: 1 }}
        componentId="mlflow.eval-runs.run-name-cell"
        id="run-name-cell"
      >
        {row.original.info.runName}
      </Typography.Link>
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

  const openDatasetDrawer = () => {
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
  const tagContent = baseTagContent;

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
      <Tooltip componentId={`mlflow.eval-runs.sortable-header-cell.tooltip-${column.id}`} content={displayedKey}>
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
  const Icon = isRowHidden(runUuid) ? VisibleOffIcon : VisibleIcon;

  return <Icon onClick={() => toggleRowVisibility(runUuid)} />;
};

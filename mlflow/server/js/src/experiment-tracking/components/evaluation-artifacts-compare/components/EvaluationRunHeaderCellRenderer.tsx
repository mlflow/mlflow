import {
  Button,
  DropdownMenu,
  OverflowIcon,
  PlayIcon,
  StopIcon,
  LegacyTooltip,
  VisibleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import ExperimentRoutes from '../../../routes';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { EvaluationRunHeaderModelIndicator } from './EvaluationRunHeaderModelIndicator';
import { shouldEnablePromptLab } from '../../../../common/utils/FeatureUtils';
import { EvaluationRunHeaderDatasetIndicator } from './EvaluationRunHeaderDatasetIndicator';
import type { RunDatasetWithTags } from '../../../types';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';
import { FormattedMessage, useIntl } from 'react-intl';
import React, { useMemo } from 'react';
import { EvaluationTableHeader } from './EvaluationTableHeader';
import { useCreateNewRun } from '../../experiment-page/hooks/useCreateNewRun';
import { canEvaluateOnRun } from '../../prompt-engineering/PromptEngineering.utils';
import { useGetExperimentRunColor } from '../../experiment-page/hooks/useExperimentRunColor';
import { RunColorPill } from '../../experiment-page/components/RunColorPill';

interface EvaluationRunHeaderCellRendererProps {
  run: RunRowType;
  onHideRun: (runUuid: string) => void;
  onDuplicateRun: (run: RunRowType) => void;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
  groupHeaderContent?: React.ReactNode;
}

/**
 * Component used as a column header for output ("run") columns
 */
export const EvaluationRunHeaderCellRenderer = ({
  run,
  onHideRun,
  onDuplicateRun,
  onDatasetSelected,
  groupHeaderContent = null,
}: EvaluationRunHeaderCellRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const { getEvaluableRowCount, evaluateAllClick, runColumnsBeingEvaluated, canEvaluateInRunColumn } =
    usePromptEngineeringContext();
  const intl = useIntl();
  const evaluableRowCount = getEvaluableRowCount(run);
  const getRunColor = useGetExperimentRunColor();
  const evaluateAllButtonEnabled = evaluableRowCount > 0;

  const evaluatingAllInProgress = runColumnsBeingEvaluated.includes(run.runUuid);

  const evaluateAllTooltipContent = useMemo(() => {
    if (!evaluateAllButtonEnabled) {
      return intl.formatMessage({
        defaultMessage: 'There are no evaluable rows within this column',
        description:
          'Experiment page > artifact compare view > run column header > Disabled "Evaluate all" button tooltip when no rows are evaluable',
      });
    }
    if (evaluateAllButtonEnabled && !evaluatingAllInProgress) {
      return intl.formatMessage(
        {
          defaultMessage: 'Process {evaluableRowCount} rows without evaluation output',
          description: 'Experiment page > artifact compare view > run column header > "Evaluate all" button tooltip',
        },
        {
          evaluableRowCount,
        },
      );
    }

    return null;
  }, [evaluableRowCount, evaluateAllButtonEnabled, evaluatingAllInProgress, intl]);

  return (
    <EvaluationTableHeader
      css={{
        justifyContent: 'flex-start',
        padding: theme.spacing.sm,
        paddingBottom: 0,
        paddingTop: theme.spacing.sm,
        flexDirection: 'column',
        gap: theme.spacing.xs / 2,
        overflow: 'hidden',
      }}
      groupHeaderContent={groupHeaderContent}
    >
      <div
        css={{
          width: '100%',
          display: 'flex',
        }}
      >
        <span css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <RunColorPill color={getRunColor(run.runUuid)} />
          <Link to={ExperimentRoutes.getRunPageRoute(run.experimentId || '', run.runUuid)} target="_blank">
            {run.runName}
          </Link>
        </span>
        <div css={{ flexBasis: theme.spacing.sm, flexShrink: 0 }} />

        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_112"
          onClick={() => onHideRun(run.runUuid)}
          size="small"
          icon={<VisibleIcon />}
          css={{ flexShrink: 0 }}
        />
        <div css={{ flex: 1 }} />
        {shouldEnablePromptLab() && canEvaluateInRunColumn(run) && (
          <>
            <div css={{ flexBasis: theme.spacing.sm }} />
            <LegacyTooltip title={evaluateAllTooltipContent}>
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_118"
                disabled={!evaluateAllButtonEnabled}
                size="small"
                onClick={() => evaluateAllClick(run)}
                icon={evaluatingAllInProgress ? <StopIcon /> : <PlayIcon />}
              >
                {evaluatingAllInProgress ? (
                  <FormattedMessage
                    defaultMessage="Stop evaluating"
                    description='Experiment page > artifact compare view > run column header > "Evaluate all" button label when the column is being evaluated'
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Evaluate all"
                    description='Experiment page > artifact compare view > run column header > "Evaluate all" button label'
                  />
                )}
              </Button>
            </LegacyTooltip>
          </>
        )}
        <div css={{ flexBasis: theme.spacing.sm }} />
        {shouldEnablePromptLab() && canEvaluateOnRun(run) && (
          <DropdownMenu.Root modal={false}>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_143"
                size="small"
                icon={<OverflowIcon />}
              />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadercellrenderer.tsx_150"
                onClick={() => onDuplicateRun(run)}
              >
                <FormattedMessage
                  defaultMessage="Duplicate run"
                  description='Experiment page > artifact compare view > run column header > "duplicate run" button label'
                />
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        )}
      </div>

      {shouldEnablePromptLab() && canEvaluateOnRun(run) ? (
        <EvaluationRunHeaderModelIndicator run={run} />
      ) : (
        <EvaluationRunHeaderDatasetIndicator run={run} onDatasetSelected={onDatasetSelected} />
      )}
    </EvaluationTableHeader>
  );
};

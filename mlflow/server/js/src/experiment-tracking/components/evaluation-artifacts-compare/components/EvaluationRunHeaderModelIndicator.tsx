import { Button, Popover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useSelector } from 'react-redux';
import type { ReduxState } from '../../../../redux-types';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import {
  canEvaluateOnRun,
  extractEvaluationPrerequisitesForRun,
} from '../../prompt-engineering/PromptEngineering.utils';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';

interface EvaluationRunHeaderModelIndicatorProps {
  run: RunRowType;
}
export const EvaluationRunHeaderModelIndicator = ({ run }: EvaluationRunHeaderModelIndicatorProps) => {
  const { theme } = useDesignSystemTheme();

  const { isHeaderExpanded } = usePromptEngineeringContext();

  const promptEvaluationDataForRun = extractEvaluationPrerequisitesForRun(run);

  const gatewayRoute = useSelector(({ modelGateway }: ReduxState) => {
    const gatewayKey = `${promptEvaluationDataForRun.routeType}:${promptEvaluationDataForRun.routeName}`;
    return promptEvaluationDataForRun.routeName ? modelGateway.modelGatewayRoutes[gatewayKey] : null;
  });

  if (!canEvaluateOnRun(run) || !promptEvaluationDataForRun) {
    return null;
  }

  const { parameters, promptTemplate, routeName } = promptEvaluationDataForRun;
  const { stop: stopSequences = [] } = parameters;

  return (
    <div
      css={{
        marginTop: theme.spacing.xs,
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        overflowX: 'hidden',
        width: '100%',
      }}
    >
      {gatewayRoute && 'mlflowDeployment' in gatewayRoute && gatewayRoute.mlflowDeployment && (
        <Typography.Hint>{gatewayRoute.mlflowDeployment.name}</Typography.Hint>
      )}
      {isHeaderExpanded && (
        <>
          <Typography.Hint>
            <FormattedMessage
              // eslint-disable-next-line formatjs/enforce-placeholders -- TODO(FEINF-2480)
              defaultMessage="Temperature: {temperature}"
              description="Experiment page > artifact compare view > run column header prompt metadata > temperature parameter"
              values={parameters}
            />
          </Typography.Hint>
          <Typography.Hint>
            <FormattedMessage
              // eslint-disable-next-line formatjs/enforce-placeholders -- TODO(FEINF-2480)
              defaultMessage="Max. tokens: {max_tokens}"
              description="Experiment page > artifact compare view > run column header prompt metadata > max tokens parameter"
              values={parameters}
            />
          </Typography.Hint>
          {stopSequences.length ? (
            <Typography.Hint css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              <FormattedMessage
                defaultMessage="Stop sequences: {stopSequences}"
                description="Experiment page > artifact compare view > run column header prompt metadata > stop sequences parameter"
                values={{ stopSequences: stopSequences?.join(', ') }}
              />
            </Typography.Hint>
          ) : null}
          <div css={{ fontSize: 0 }}>
            <Popover.Root componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadermodelindicator.tsx_107">
              <Popover.Trigger asChild>
                <Button
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationrunheadermodelindicator.tsx_115"
                  type="link"
                  size="small"
                  css={{
                    fontSize: theme.typography.fontSizeSm,
                  }}
                >
                  <FormattedMessage
                    defaultMessage="View prompt template"
                    description='Experiment page > artifact compare view > run column header prompt metadata > "view prompt template" button label'
                  />
                </Button>
              </Popover.Trigger>
              <Popover.Content css={{ maxWidth: 300 }}>
                <Popover.Arrow />
                {promptTemplate}
              </Popover.Content>
            </Popover.Root>
          </div>
        </>
      )}
    </div>
  );
};

import { Button, Popover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useSelector } from 'react-redux';
import { ReduxState } from '../../../../redux-types';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import {
  canEvaluateOnRun,
  extractEvaluationPrerequisitesForRun,
} from '../../prompt-engineering/PromptEngineering.utils';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';

interface EvaluationRunHeaderModelIndicatorProps {
  run: RunRowType;
}
export const EvaluationRunHeaderModelIndicator = ({
  run,
}: EvaluationRunHeaderModelIndicatorProps) => {
  const { theme } = useDesignSystemTheme();

  const { isHeaderExpanded } = usePromptEngineeringContext();

  const promptEvaluationDataForRun = extractEvaluationPrerequisitesForRun(run);

  const gatewayRoute = useSelector(({ modelGateway }: ReduxState) =>
    promptEvaluationDataForRun.routeName
      ? modelGateway.modelGatewayRoutes[promptEvaluationDataForRun.routeName]
      : null,
  );

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
      <Typography.Hint>{gatewayRoute ? gatewayRoute.model.name : routeName}</Typography.Hint>
      {isHeaderExpanded && (
        <>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage='Temperature: {temperature}'
              description='Experiment page > artifact compare view > run column header prompt metadata > temperature parameter'
              values={parameters}
            />
          </Typography.Hint>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage='Max. tokens: {max_tokens}'
              description='Experiment page > artifact compare view > run column header prompt metadata > max tokens parameter'
              values={parameters}
            />
          </Typography.Hint>
          {stopSequences.length ? (
            <Typography.Hint
              css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            >
              <FormattedMessage
                defaultMessage='Stop sequences: {stopSequences}'
                description='Experiment page > artifact compare view > run column header prompt metadata > stop sequences parameter'
                values={{ stopSequences: stopSequences?.join(', ') }}
              />
            </Typography.Hint>
          ) : null}
          <div css={{ fontSize: 0 }}>
            <Popover.Root>
              <Popover.Trigger asChild>
                <Button
                  type='link'
                  size='small'
                  css={{
                    fontSize: theme.typography.fontSizeSm,
                  }}
                >
                  <FormattedMessage
                    defaultMessage='View prompt template'
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

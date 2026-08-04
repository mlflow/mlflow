import { Button, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCallback, useState } from 'react';

import { AssistantSparkleIcon, useAssistant } from '@mlflow/mlflow/src/assistant';
import { buildAnalyzeEvaluationRunAssistantPrompt } from '../../pages/experiment-evaluation-runs/evalRunsAgentPrompt';
import { getAiGradientBorderStyle } from '@mlflow/mlflow/src/shared/web-shared/design-system/aiGradientBorderStyle';

export const RunViewEvaluationAnalyzeButton = ({ runUuid }: { runUuid: string }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { openPanel, prefillPrompt, canUseAssistant } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  const onAnalyzeClick = useCallback(() => {
    openPanel();
    prefillPrompt(buildAnalyzeEvaluationRunAssistantPrompt(runUuid));
  }, [openPanel, prefillPrompt, runUuid]);

  if (!canUseAssistant) {
    return null;
  }

  return (
    <Tooltip
      componentId="mlflow.run-view-evaluations.analyze-button.tooltip"
      content={intl.formatMessage({
        defaultMessage: 'Analyze evaluation run with Assistant',
        description: 'Tooltip for the assistant analyze button in the evaluation run toolbar',
      })}
    >
      <Button
        componentId="mlflow.run-view-evaluations.analyze-button"
        onClick={onAnalyzeClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        css={{
          ...getAiGradientBorderStyle(theme),
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 32,
          padding: `0 ${theme.spacing.sm}px`,
          lineHeight: theme.typography.lineHeightBase,
        }}
      >
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: theme.spacing.xs,
            lineHeight: theme.typography.lineHeightBase,
          }}
        >
          <AssistantSparkleIcon isHovered={isHovered} iconSize={16} />
          <span css={{ display: 'inline-flex', alignItems: 'center' }}>
            <FormattedMessage
              defaultMessage="Analyze"
              description="Button label for analyzing an evaluation run with Assistant"
            />
          </span>
        </span>
      </Button>
    </Tooltip>
  );
};

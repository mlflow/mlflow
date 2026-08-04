import { Button, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCallback } from 'react';

import { AssistantSparkleIcon, useAssistant } from '@mlflow/mlflow/src/assistant';
import { buildAnalyzeEvaluationRunAssistantPrompt } from '../../pages/experiment-evaluation-runs/evalRunsAgentPrompt';

export const RunViewEvaluationAnalyzeButton = ({ runUuid }: { runUuid: string }) => {
  const intl = useIntl();
  const { openPanel, prefillPrompt, canUseAssistant } = useAssistant();

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
        icon={<AssistantSparkleIcon isHovered={false} iconSize={16} />}
        onClick={onAnalyzeClick}
      >
        <FormattedMessage
          defaultMessage="Analyze"
          description="Button label for analyzing an evaluation run with Assistant"
        />
      </Button>
    </Tooltip>
  );
};

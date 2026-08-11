import { useState } from 'react';
import { Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useAssistant, AssistantSparkleIcon } from '@mlflow/mlflow/src/assistant';

interface EvalRunsAnalyzeButtonProps {
  runName?: string;
  comparedToRunName?: string;
}

export const EvalRunsAnalyzeButton = ({ runName, comparedToRunName }: EvalRunsAnalyzeButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { openPanel, prefillPrompt } = useAssistant();
  const [isHovered, setIsHovered] = useState(false);

  // Only render if both run names are present
  if (!runName || !comparedToRunName) {
    return null;
  }

  const handleClick = () => {
    const prompt = intl.formatMessage(
      {
        defaultMessage:
          'Compare evaluation run "{runName}" against "{comparedToRunName}". Explain what changed and why the scores moved. Then recommend exactly one of: promote this to baseline, hold off and gather more data, the result is ambiguous, or this is a clear regression. Justify the recommendation in two sentences.',
        description: 'Prompt for Assistant to analyze evaluation run comparison',
      },
      {
        runName,
        comparedToRunName,
      },
    );

    prefillPrompt(prompt);
    openPanel();
  };

  return (
    <Button
      componentId="mlflow.eval-runs.analyze-button"
      type="tertiary"
      onClick={handleClick}
      icon={<AssistantSparkleIcon isHovered={isHovered} />}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      css={{
        marginLeft: 'auto',
      }}
    >
      <FormattedMessage defaultMessage="Analyze" description="Button label to analyze evaluation run comparison" />
    </Button>
  );
};

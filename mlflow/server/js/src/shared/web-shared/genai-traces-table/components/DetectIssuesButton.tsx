import React from 'react';
import { Button, SparkleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface DetectIssuesButtonProps {
  componentId: string;
  onClick: () => void;
}

export const DetectIssuesButton: React.FC<DetectIssuesButtonProps> = ({ componentId, onClick }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Button
      componentId={componentId}
      onClick={onClick}
      aria-label={intl.formatMessage({
        defaultMessage: 'Detect issues in traces',
        description: 'Aria label for the detect issues button',
      })}
      icon={<SparkleIcon color="ai" />}
      css={{
        border: '1px solid transparent !important',
        background: `linear-gradient(${theme.colors.backgroundPrimary}, ${theme.colors.backgroundPrimary}) padding-box, ${theme.gradients.aiBorderGradient} border-box`,
      }}
    >
      <FormattedMessage defaultMessage="Detect Issues" description="Label for the detect issues button" />
    </Button>
  );
};

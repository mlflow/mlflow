import { Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

type ComponentIds = 'mlflow.gateway.edit-endpoint.prompt-caching' | 'mlflow.gateway.create-endpoint.prompt-caching';

export interface PromptCachingConfiguratorProps {
  value: boolean;
  onChange: (value: boolean) => void;
  componentId?: ComponentIds;
}

export const PromptCachingConfigurator = ({
  value,
  onChange,
  componentId = 'mlflow.gateway.edit-endpoint.prompt-caching',
}: PromptCachingConfiguratorProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Switch
          componentId={`${componentId}.toggle`}
          checked={value}
          onChange={(checked) => onChange(checked)}
          aria-label="Enable prompt caching"
        />
        <Typography.Text>
          <FormattedMessage defaultMessage="Enable prompt caching" description="Label for prompt caching toggle" />
        </Typography.Text>
      </div>

      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        <FormattedMessage
          defaultMessage="When enabled, repeated prompts are cached by the provider to reduce latency and cost. Currently supported for Mistral endpoints."
          description="Prompt caching description"
        />
      </Typography.Text>
    </div>
  );
};

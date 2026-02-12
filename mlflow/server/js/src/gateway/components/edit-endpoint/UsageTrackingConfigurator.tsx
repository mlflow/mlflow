import { Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

type ComponentIds = 'mlflow.gateway.edit-endpoint.usage-tracking' | 'mlflow.gateway.create-endpoint.usage-tracking';

export interface UsageTrackingConfiguratorProps {
  value: boolean;
  onChange: (value: boolean) => void;
  componentIdPrefix?: ComponentIds;
}

export const UsageTrackingConfigurator = ({
  value,
  onChange,
  componentIdPrefix = 'mlflow.gateway.edit-endpoint.usage-tracking',
}: UsageTrackingConfiguratorProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Switch
          componentId={`${componentIdPrefix}.toggle`}
          checked={value}
          onChange={(checked) => onChange(checked)}
          aria-label="Enable usage tracking"
        />
        <Typography.Text>
          <FormattedMessage defaultMessage="Enable usage tracking" description="Label for usage tracking toggle" />
        </Typography.Text>
      </div>

      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        <FormattedMessage
          defaultMessage="When enabled, all requests to this endpoint will be logged as traces in an MLflow experiment. This allows you to monitor usage, debug issues, and analyze performance."
          description="Usage tracking description"
        />
      </Typography.Text>
    </div>
  );
};

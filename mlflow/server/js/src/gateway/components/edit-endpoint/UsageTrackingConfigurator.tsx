import { Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ExperimentSelect } from '../shared/ExperimentSelect';

type ComponentIds = 'mlflow.gateway.edit-endpoint.usage-tracking' | 'mlflow.gateway.create-endpoint.usage-tracking';

export interface UsageTrackingConfiguratorProps {
  value: boolean;
  onChange: (value: boolean) => void;
  experimentId: string;
  onExperimentIdChange: (experimentId: string) => void;
  componentIdPrefix?: ComponentIds;
}

export const UsageTrackingConfigurator = ({
  value,
  onChange,
  experimentId,
  onExperimentIdChange,
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

      {value && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold css={{ fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="Experiment" description="Label for experiment selector" />
          </Typography.Text>
          <ExperimentSelect
            value={experimentId}
            onChange={onExperimentIdChange}
            componentIdPrefix={`${componentIdPrefix}.experiment`}
          />
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage
              defaultMessage="Select an existing experiment or leave blank to auto-create one named 'gateway/[endpoint_name]'."
              description="Experiment selector help text"
            />
          </Typography.Text>
        </div>
      )}
    </div>
  );
};

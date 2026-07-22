import { Radio, Typography, useDesignSystemTheme, type RadioChangeEvent } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

type ComponentIds = 'mlflow.gateway.edit-endpoint.usage-tracking' | 'mlflow.gateway.create-endpoint.usage-tracking';

/**
 * The three valid tracing states for an endpoint, combining the `usage_tracking`
 * and `exclude_content` flags ("exclude content without tracking" is meaningless):
 * - 'off': no traces are logged
 * - 'metadata_only': traces are logged without request/response content
 * - 'full': traces are logged with complete request/response content
 */
export type UsageTrackingMode = 'off' | 'metadata_only' | 'full';

export const getUsageTrackingMode = (usageTracking: boolean, excludeContent: boolean): UsageTrackingMode => {
  if (!usageTracking) {
    return 'off';
  }
  return excludeContent ? 'metadata_only' : 'full';
};

export interface UsageTrackingConfiguratorProps {
  mode: UsageTrackingMode;
  onChange: (mode: UsageTrackingMode) => void;
  disabled?: boolean;
  /** Hide the per-option descriptions for compact layouts (e.g. the edit sidebar) */
  compact?: boolean;
  componentId?: ComponentIds;
}

export const UsageTrackingConfigurator = ({
  mode,
  onChange,
  disabled,
  compact,
  componentId = 'mlflow.gateway.edit-endpoint.usage-tracking',
}: UsageTrackingConfiguratorProps) => {
  const { theme } = useDesignSystemTheme();

  const renderDescription = (content: React.ReactNode) =>
    compact ? null : (
      <Typography.Text
        color="secondary"
        css={{ display: 'block', fontSize: theme.typography.fontSizeSm, marginLeft: theme.spacing.lg }}
      >
        {content}
      </Typography.Text>
    );

  return (
    <Radio.Group
      componentId={`${componentId}.mode`}
      name={`${componentId}.mode`}
      value={mode}
      onChange={(e: RadioChangeEvent) => onChange(e.target.value as UsageTrackingMode)}
      disabled={disabled}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: compact ? 0 : theme.spacing.sm }}>
        <div>
          <Radio value="off" aria-label="Tracing off">
            <FormattedMessage defaultMessage="Off" description="Usage tracking mode: tracing disabled" />
          </Radio>
          {renderDescription(
            <FormattedMessage
              defaultMessage="Requests to this endpoint are not logged as traces."
              description="Description for tracing off mode"
            />,
          )}
        </div>
        <div>
          <Radio value="metadata_only" aria-label="Redact message content">
            <FormattedMessage
              defaultMessage="Redact message content"
              description="Usage tracking mode: message content redacted"
            />
          </Radio>
          {renderDescription(
            <FormattedMessage
              defaultMessage="Requests are logged as traces with usage metadata such as token consumption, latency, and status. Prompts, messages, and model responses are redacted."
              description="Description for redacted message content tracing mode"
            />,
          )}
        </div>
        <div>
          <Radio value="full" aria-label="Full tracing">
            <FormattedMessage defaultMessage="Full" description="Usage tracking mode: full tracing" />
          </Radio>
          {renderDescription(
            <FormattedMessage
              defaultMessage="Requests are logged as traces with complete request and response content. This allows you to monitor usage, debug issues, and analyze performance."
              description="Description for full tracing mode"
            />,
          )}
        </div>
      </div>
    </Radio.Group>
  );
};

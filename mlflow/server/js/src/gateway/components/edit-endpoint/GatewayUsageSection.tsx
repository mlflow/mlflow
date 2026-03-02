import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { GatewayChartsPanel } from '../GatewayChartsPanel';

interface GatewayUsageSectionProps {
  experimentId: string;
  tooltipLinkUrlBuilder?: (experimentId: string, timestampMs: number, timeIntervalSeconds: number) => string;
}

export const GatewayUsageSection = ({ experimentId, tooltipLinkUrlBuilder }: GatewayUsageSectionProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div>
      <div css={{ marginBottom: theme.spacing.md }}>
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Usage" description="Section title for endpoint usage" />
        </Typography.Title>
        <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Monitor endpoint usage and performance metrics"
            description="Usage section description"
          />
        </Typography.Text>
        <Typography.Link
          componentId="mlflow.gateway.endpoint.usage.view-full-dashboard"
          href={`#/experiments/${experimentId}/overview`}
          css={{ display: 'inline-block', marginTop: theme.spacing.xs }}
        >
          <FormattedMessage defaultMessage="View full dashboard" description="Link to view full usage dashboard" />
        </Typography.Link>
      </div>

      <GatewayChartsPanel
        experimentIds={[experimentId]}
        showTokenStats
        tooltipLinkUrlBuilder={tooltipLinkUrlBuilder}
        tooltipLinkText={
          tooltipLinkUrlBuilder ? (
            <FormattedMessage
              defaultMessage="View logs for this period"
              description="Link text to navigate to gateway endpoint logs tab"
            />
          ) : undefined
        }
      />
    </div>
  );
};

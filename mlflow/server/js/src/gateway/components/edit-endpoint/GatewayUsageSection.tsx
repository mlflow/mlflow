import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { GatewayChartsPanel } from '../GatewayChartsPanel';

interface GatewayUsageSectionProps {
  experimentId: string;
}

export const GatewayUsageSection = ({ experimentId }: GatewayUsageSectionProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div>
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.md }}
      >
        <div>
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Usage" description="Section title for endpoint usage" />
          </Typography.Title>
          <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Monitor endpoint usage and performance metrics"
              description="Usage section description"
            />
          </Typography.Text>
        </div>
        <Typography.Link
          componentId="mlflow.gateway.endpoint.usage.view-full-dashboard"
          href={`#/experiments/${experimentId}/overview`}
          css={{ fontSize: theme.typography.fontSizeSm }}
        >
          <FormattedMessage defaultMessage="View full dashboard" description="Link to view full usage dashboard" />
        </Typography.Link>
      </div>

      <GatewayChartsPanel experimentIds={[experimentId]} showTokenStats showCostCharts />
    </div>
  );
};

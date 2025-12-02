import { Drawer, Empty, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { timestampToDate } from '../../utils/dateUtils';
import GatewayRoutes from '../../routes';
import type { Endpoint } from '../../types';

interface EndpointsUsingModelDrawerProps {
  open: boolean;
  endpoints: Endpoint[];
  onClose: () => void;
}

export const EndpointsUsingModelDrawer = ({ open, endpoints, onClose }: EndpointsUsingModelDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const { formatDate } = useIntl();

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return formatDate(timestampToDate(timestamp), {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      timeZoneName: 'short',
    });
  };

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.endpoints-using-model.drawer"
        width={480}
        title={
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="Endpoints ({count})"
              description="Title for endpoints using model drawer"
              values={{ count: endpoints.length }}
            />
          </Typography.Title>
        }
      >
        <Spacer size="md" />
        {endpoints.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No endpoints are using this model"
                description="Empty state when no endpoints use the model"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {endpoints.map((endpoint) => (
              <div
                key={endpoint.endpoint_id}
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.xs,
                  padding: theme.spacing.md,
                  border: `1px solid ${theme.colors.borderDecorative}`,
                  borderRadius: theme.general.borderRadiusBase,
                }}
              >
                <Link
                  to={GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id)}
                  css={{
                    color: theme.colors.actionPrimaryBackgroundDefault,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                    textDecoration: 'none',
                    '&:hover': {
                      textDecoration: 'underline',
                    },
                  }}
                >
                  {endpoint.name}
                </Link>
                <span css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
                  <FormattedMessage
                    defaultMessage="Added on {date}"
                    description="Date when endpoint was created"
                    values={{ date: formatTimestamp(endpoint.created_at) }}
                  />
                </span>
              </div>
            ))}
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};

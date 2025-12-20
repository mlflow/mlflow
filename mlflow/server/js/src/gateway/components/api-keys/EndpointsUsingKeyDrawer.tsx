import { Drawer, Empty, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { timestampToDate } from '../../utils/dateUtils';
import GatewayRoutes from '../../routes';
import type { Endpoint } from '../../types';

interface EndpointsUsingKeyDrawerProps {
  open: boolean;
  keyName: string;
  endpoints: Endpoint[];
  onClose: () => void;
}

export const EndpointsUsingKeyDrawer = ({ open, keyName, endpoints, onClose }: EndpointsUsingKeyDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
    }
  };

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.endpoints-using-key.drawer"
        width={480}
        title={
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="Endpoints ({count})"
              description="Gateway > Endpoints using key drawer > Title"
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
                defaultMessage="No endpoints are using this key"
                description="Gateway > Endpoints using key drawer > Empty state"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Endpoints using key: {name}"
                description="Gateway > Endpoints using key drawer > Subtitle showing key name"
                values={{ name: keyName }}
              />
            </Typography.Text>

            <div
              css={{
                border: `1px solid ${theme.colors.borderDecorative}`,
                borderRadius: theme.general.borderRadiusBase,
                overflow: 'hidden',
              }}
            >
              {endpoints.map((endpoint, index) => (
                <div
                  key={endpoint.endpoint_id}
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.xs,
                    padding: theme.spacing.sm,
                    borderBottom: index < endpoints.length - 1 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                  }}
                >
                  <Link
                    to={GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id)}
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      textDecoration: 'none',
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      '&:hover': {
                        textDecoration: 'underline',
                      },
                    }}
                  >
                    {endpoint.name || endpoint.endpoint_id}
                  </Link>
                  {endpoint.model_mappings && endpoint.model_mappings.length > 0 && (
                    <div
                      css={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: theme.spacing.xs,
                        color: theme.colors.textSecondary,
                        fontSize: theme.typography.fontSizeSm,
                      }}
                    >
                      {endpoint.model_mappings.map((mapping, mappingIndex) => (
                        <span key={mapping.mapping_id}>
                          {mapping.model_definition?.model_name ?? 'Unknown model'}
                          {mappingIndex < endpoint.model_mappings.length - 1 && ','}
                        </span>
                      ))}
                    </div>
                  )}
                  <span css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
                    <FormattedMessage
                      defaultMessage="Created {date}"
                      description="Gateway > Endpoints using key drawer > Endpoint created date"
                      values={{
                        date: intl.formatDate(timestampToDate(endpoint.created_at), {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                        }),
                      }}
                    />
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};

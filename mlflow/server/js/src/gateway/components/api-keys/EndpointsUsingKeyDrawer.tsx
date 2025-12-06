import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { timestampToDate } from '../../utils/dateUtils';
import GatewayRoutes from '../../routes';
import { RelationshipListDrawer } from '../common/RelationshipListDrawer';
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

  return (
    <RelationshipListDrawer
      open={open}
      onClose={onClose}
      componentId="mlflow.gateway.endpoints-using-key.drawer"
      title={
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="Endpoints ({count})"
            description="Gateway > Endpoints using key drawer > Title"
            values={{ count: endpoints.length }}
          />
        </Typography.Title>
      }
      subtitle={
        <FormattedMessage
          defaultMessage="Endpoints using key: {name}"
          description="Gateway > Endpoints using key drawer > Subtitle showing key name"
          values={{ name: keyName }}
        />
      }
      emptyMessage={
        <FormattedMessage
          defaultMessage="No endpoints are using this key"
          description="Gateway > Endpoints using key drawer > Empty state"
        />
      }
      sections={[
        {
          key: 'endpoints',
          title: (
            <FormattedMessage
              defaultMessage="Endpoints ({count})"
              description="Gateway > Endpoints using key drawer > Endpoints section header"
              values={{ count: endpoints.length }}
            />
          ),
          items: endpoints,
          renderItem: (endpoint) => (
            <>
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
                {endpoint.name ?? endpoint.endpoint_id}
              </Link>
              <div
                css={{
                  display: 'flex',
                  gap: theme.spacing.md,
                  color: theme.colors.textSecondary,
                  fontSize: theme.typography.fontSizeSm,
                }}
              >
                <span>
                  <FormattedMessage
                    defaultMessage="{count, plural, one {# model} other {# models}}"
                    description="Gateway > Endpoints using key drawer > Model count"
                    values={{ count: endpoint.model_mappings?.length ?? 0 }}
                  />
                </span>
              </div>
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
            </>
          ),
        },
      ]}
    />
  );
};

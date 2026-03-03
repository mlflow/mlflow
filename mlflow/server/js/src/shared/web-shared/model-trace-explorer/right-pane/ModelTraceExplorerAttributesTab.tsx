import { isNil, isString, keys } from 'lodash';

import { Empty, NewWindowIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { SPAN_ATTRIBUTE_LINKED_GATEWAY_TRACE_ID } from '../constants';
import { useGatewayTraceLink } from '../hooks/useGatewayTraceLink';
import { Link } from '../RoutingUtils';

export function ModelTraceExplorerAttributesTab({
  activeSpan,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { attributes } = activeSpan;
  const containsAttributes = keys(attributes).length > 0;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  const linkedTraceId = !Array.isArray(attributes)
    ? attributes?.[SPAN_ATTRIBUTE_LINKED_GATEWAY_TRACE_ID]
    : undefined;
  const gatewayTraceHref = useGatewayTraceLink(isString(linkedTraceId) ? linkedTraceId : undefined);

  if (!containsAttributes || isNil(attributes)) {
    return (
      <div css={{ marginTop: theme.spacing.md }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No attributes found"
              description="Empty state for the attributes tab in the model trace explorer. Attributes are properties of a span that the user defines."
            />
          }
        />
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
      }}
    >
      {Object.entries(attributes).map(([key, value]) => {
        if (key === SPAN_ATTRIBUTE_LINKED_GATEWAY_TRACE_ID && gatewayTraceHref) {
          return (
            <div
              key={key}
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                border: `1px solid ${theme.colors.border}`,
                overflow: 'hidden',
              }}
            >
              <div css={{ padding: theme.spacing.sm }}>
                <Typography.Title level={4} color="secondary" withoutMargins>
                  {key}
                </Typography.Title>
              </div>
              <div
                css={{
                  padding: theme.spacing.sm,
                  borderTop: `1px solid ${theme.colors.border}`,
                  backgroundColor: theme.colors.backgroundSecondary,
                }}
              >
                <Link to={gatewayTraceHref} target="_blank" rel="noreferrer">
                  <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                    <FormattedMessage
                      defaultMessage="View gateway trace"
                      description="Link text for navigating to the corresponding gateway trace"
                    />
                    <NewWindowIcon css={{ fontSize: 12 }} />
                  </span>
                </Link>
              </div>
            </div>
          );
        }

        return (
          <ModelTraceExplorerCodeSnippet
            key={key}
            title={key}
            data={JSON.stringify(value, null, 2)}
            searchFilter={searchFilter}
            activeMatch={activeMatch}
            containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'attributes' && activeMatch.key === key}
          />
        );
      })}
    </div>
  );
}

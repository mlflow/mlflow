import { useCallback, useMemo, useState } from 'react';
import { Accordion, ChevronRightIcon, importantify, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { EndpointBinding, ResourceType } from '../../types';

interface ConnectedResourcesSectionProps {
  bindings: EndpointBinding[];
}

export const ConnectedResourcesSection = ({ bindings }: ConnectedResourcesSectionProps) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const intl = useIntl();

  const resourceTypes = useMemo(() => Array.from(new Set(bindings.map((b) => b.resource_type))), [bindings]);

  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());

  const expandedSections = useMemo(
    () => resourceTypes.filter((rt) => !collapsedSections.has(rt)),
    [resourceTypes, collapsedSections],
  );

  const formatResourceTypePlural = (type: string) => {
    switch (type) {
      case 'scorer_job':
        return intl.formatMessage({ defaultMessage: 'Scorer jobs', description: 'Scorer jobs resource type plural' });
      default:
        return type;
    }
  };

  const getExpandIcon = useCallback(
    ({ isActive }: { isActive?: boolean }) => (
      <div
        css={importantify({
          width: theme.general.heightBase / 2,
          transform: isActive ? 'rotate(90deg)' : undefined,
          transition: 'transform 0.2s',
        })}
      >
        <ChevronRightIcon
          css={{
            svg: { width: theme.general.heightBase / 2, height: theme.general.heightBase / 2 },
          }}
        />
      </div>
    ),
    [theme],
  );

  const accordionStyles = useMemo(() => {
    const clsPrefix = getPrefixedClassName('collapse');
    const classItem = `.${clsPrefix}-item`;
    const classHeader = `.${clsPrefix}-header`;
    const classContentBox = `.${clsPrefix}-content-box`;

    return {
      border: 'none',
      backgroundColor: 'transparent',
      [`& > ${classItem}`]: {
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        marginBottom: theme.spacing.xs,
        overflow: 'hidden',
      },
      [`& > ${classItem} > ${classHeader}`]: {
        paddingLeft: 0,
        paddingTop: theme.spacing.xs,
        paddingBottom: theme.spacing.xs,
        display: 'flex',
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
      },
      [classContentBox]: {
        padding: 0,
      },
    };
  }, [theme, getPrefixedClassName]);

  const bindingsByType = useMemo(() => {
    const groups = new Map<ResourceType, EndpointBinding[]>();
    bindings.forEach((binding) => {
      if (!groups.has(binding.resource_type)) {
        groups.set(binding.resource_type, []);
      }
      groups.get(binding.resource_type)!.push(binding);
    });
    return groups;
  }, [bindings]);

  const handleAccordionChange = useCallback(
    (keys: string | string[]) => {
      const expandedKeys = new Set(Array.isArray(keys) ? keys : [keys]);
      setCollapsedSections(new Set(resourceTypes.filter((rt) => !expandedKeys.has(rt))));
    },
    [resourceTypes],
  );

  return (
    <div>
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Used by" description="Used by label for connected resources" />
      </Typography.Text>
      <div css={{ marginTop: theme.spacing.xs }}>
        {bindings.length === 0 ? (
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, fontStyle: 'italic' }}>
            <FormattedMessage
              defaultMessage="No resources are using this endpoint"
              description="Empty state for connected resources"
            />
          </Typography.Text>
        ) : (
          <Accordion
            componentId="mlflow.gateway.endpoint-details.bindings-accordion"
            activeKey={expandedSections}
            onChange={handleAccordionChange}
            dangerouslyAppendEmotionCSS={accordionStyles}
            dangerouslySetAntdProps={{
              expandIconPosition: 'left',
              expandIcon: getExpandIcon,
            }}
          >
            {Array.from(bindingsByType.entries()).map(([resourceType, typeBindings]) => (
              <Accordion.Panel
                key={resourceType}
                header={
                  <span
                    css={{
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      fontSize: theme.typography.fontSizeSm,
                    }}
                  >
                    {formatResourceTypePlural(resourceType)} ({typeBindings.length})
                  </span>
                }
              >
                <div
                  css={{
                    maxHeight: 8 * 28,
                    overflowY: 'auto',
                  }}
                >
                  {typeBindings.map((binding) => (
                    <div
                      key={`${binding.endpoint_id}_${binding.resource_type}_${binding.resource_id}`}
                      css={{
                        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                        paddingLeft: theme.spacing.md,
                        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                        '&:last-child': { borderBottom: 'none' },
                      }}
                    >
                      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm, fontFamily: 'monospace' }}>
                        {binding.resource_id}
                      </Typography.Text>
                    </div>
                  ))}
                </div>
              </Accordion.Panel>
            ))}
          </Accordion>
        )}
      </div>
    </div>
  );
};

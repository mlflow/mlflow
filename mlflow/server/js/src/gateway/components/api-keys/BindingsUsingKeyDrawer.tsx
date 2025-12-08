import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Accordion,
  Checkbox,
  ChevronRightIcon,
  Drawer,
  Empty,
  Spacer,
  Typography,
  importantify,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { timestampToDate } from '../../utils/dateUtils';
import GatewayRoutes from '../../routes';
import type { Endpoint, EndpointBinding, ResourceType } from '../../types';

interface BindingsUsingKeyDrawerProps {
  open: boolean;
  bindings: EndpointBinding[];
  endpoints: Endpoint[];
  onClose: () => void;
}

export const BindingsUsingKeyDrawer = ({ open, bindings, endpoints, onClose }: BindingsUsingKeyDrawerProps) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const { formatDate, formatMessage } = useIntl();
  const [selectedResourceTypes, setSelectedResourceTypes] = useState<Set<ResourceType>>(new Set());
  const [expandedSections, setExpandedSections] = useState<string[]>([]);

  // Helper to look up endpoint name by ID
  const getEndpointName = useCallback(
    (endpointId: string) => endpoints.find((e) => e.endpoint_id === endpointId)?.name,
    [endpoints],
  );

  // Custom expand icon for accordion
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

  // Accordion styles
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
        marginBottom: theme.spacing.sm,
        overflow: 'hidden',
      },
      [`& > ${classItem} > ${classHeader}`]: {
        paddingLeft: theme.spacing.sm,
        paddingTop: theme.spacing.sm,
        paddingBottom: theme.spacing.sm,
        display: 'flex',
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
      },
      [classContentBox]: {
        padding: theme.spacing.sm,
      },
    };
  }, [theme, getPrefixedClassName]);

  // Get unique resource types from bindings
  const availableResourceTypes = useMemo(() => {
    const types = new Set<ResourceType>();
    bindings.forEach((binding) => types.add(binding.resource_type));
    return Array.from(types);
  }, [bindings]);

  // Initialize selected types and expanded sections when drawer opens
  useEffect(() => {
    if (open && bindings.length > 0) {
      const types = Array.from(new Set(bindings.map((b) => b.resource_type)));
      setSelectedResourceTypes(new Set(types));
      setExpandedSections(types);
    }
  }, [open, bindings]);

  // Group bindings by resource type
  const groupedBindings = useMemo(() => {
    const groups = new Map<ResourceType, EndpointBinding[]>();
    bindings.forEach((binding) => {
      if (!groups.has(binding.resource_type)) {
        groups.set(binding.resource_type, []);
      }
      groups.get(binding.resource_type)!.push(binding);
    });
    return groups;
  }, [bindings]);

  // Filter bindings based on selected resource types
  const filteredBindings = useMemo(() => {
    if (selectedResourceTypes.size === 0) return bindings;
    return bindings.filter((binding) => selectedResourceTypes.has(binding.resource_type));
  }, [bindings, selectedResourceTypes]);

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
      // Reset state when closing
      setSelectedResourceTypes(new Set());
      setExpandedSections([]);
    }
  };

  const handleResourceTypeToggle = (resourceType: ResourceType) => {
    setSelectedResourceTypes((prev) => {
      const next = new Set(prev);
      if (next.has(resourceType)) {
        next.delete(resourceType);
      } else {
        next.add(resourceType);
      }
      return next;
    });
  };

  const handleAccordionChange = (keys: string | string[]) => {
    setExpandedSections(Array.isArray(keys) ? keys : [keys]);
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

  // Parse resource type from snake_case to Title Case (e.g., "scorer_job" -> "Scorer Job")
  const formatResourceType = (resourceType: string, plural = false) => {
    const formatted = resourceType
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    return plural ? `${formatted}s` : formatted;
  };

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.bindings-using-key.drawer"
        width={480}
        title={
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="Used by ({count})"
              description="Title for bindings using key drawer"
              values={{ count: filteredBindings.length }}
            />
          </Typography.Title>
        }
      >
        <Spacer size="md" />
        {bindings.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No resources are using this key"
                description="Empty state when no resources use the key"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {/* Filter checkboxes */}
            {availableResourceTypes.length > 1 && (
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.sm,
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.general.borderRadiusBase,
                }}
              >
                <Typography.Text css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
                  <FormattedMessage defaultMessage="Filter by type" description="Filter by resource type label" />
                </Typography.Text>
                <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
                  {availableResourceTypes.map((resourceType) => (
                    <Checkbox
                      key={resourceType}
                      componentId={`mlflow.gateway.bindings-using-key.filter.${resourceType}`}
                      isChecked={selectedResourceTypes.has(resourceType)}
                      onChange={() => handleResourceTypeToggle(resourceType)}
                    >
                      {formatResourceType(resourceType, true)} ({groupedBindings.get(resourceType)?.length ?? 0})
                    </Checkbox>
                  ))}
                </div>
              </div>
            )}

            {/* Grouped bindings in accordion */}
            {filteredBindings.length === 0 ? (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No resources match the selected filters"
                    description="Empty state when filter returns no results"
                  />
                }
              />
            ) : (
              <Accordion
                componentId="mlflow.gateway.bindings-using-key.accordion"
                activeKey={expandedSections}
                onChange={handleAccordionChange}
                dangerouslyAppendEmotionCSS={accordionStyles}
                dangerouslySetAntdProps={{
                  expandIconPosition: 'left',
                  expandIcon: getExpandIcon,
                }}
              >
                {availableResourceTypes
                  .filter((resourceType) => selectedResourceTypes.has(resourceType))
                  .map((resourceType) => {
                    const typeBindings = groupedBindings.get(resourceType) ?? [];
                    return (
                      <Accordion.Panel
                        key={resourceType}
                        header={
                          <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
                            {formatResourceType(resourceType, true)} ({typeBindings.length})
                          </span>
                        }
                      >
                        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                          {typeBindings.map((binding) => (
                            <div
                              key={binding.binding_id}
                              css={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: theme.spacing.xs,
                                padding: theme.spacing.md,
                                border: `1px solid ${theme.colors.borderDecorative}`,
                                borderRadius: theme.general.borderRadiusBase,
                              }}
                            >
                              <span
                                css={{
                                  color: theme.colors.textPrimary,
                                  fontWeight: theme.typography.typographyBoldFontWeight,
                                }}
                              >
                                {formatResourceType(binding.resource_type)} #{binding.resource_id}
                              </span>
                              {getEndpointName(binding.endpoint_id) && (
                                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                                  <span
                                    css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}
                                  >
                                    <FormattedMessage defaultMessage="via" description="Via label for endpoint" />
                                  </span>
                                  <Link
                                    to={GatewayRoutes.getEndpointDetailsRoute(binding.endpoint_id)}
                                    css={{
                                      color: theme.colors.actionPrimaryBackgroundDefault,
                                      fontSize: theme.typography.fontSizeSm,
                                      textDecoration: 'none',
                                      '&:hover': {
                                        textDecoration: 'underline',
                                      },
                                    }}
                                  >
                                    {getEndpointName(binding.endpoint_id)}
                                  </Link>
                                </div>
                              )}
                              <span css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
                                <FormattedMessage
                                  defaultMessage="Created on {date}"
                                  description="Date when binding was created"
                                  values={{ date: formatTimestamp(binding.created_at) }}
                                />
                              </span>
                            </div>
                          ))}
                        </div>
                      </Accordion.Panel>
                    );
                  })}
              </Accordion>
            )}
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};

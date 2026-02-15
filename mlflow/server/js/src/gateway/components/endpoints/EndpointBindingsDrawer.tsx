import { useCallback, useMemo } from 'react';
import {
  Accordion,
  ChevronRightIcon,
  Drawer,
  Empty,
  importantify,
  Spacer,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { EndpointBinding, ResourceType } from '../../types';

interface EndpointBindingsDrawerProps {
  open: boolean;
  endpointName: string;
  bindings: EndpointBinding[];
  onClose: () => void;
}

const formatResourceTypePlural = (resourceType: ResourceType): string => {
  switch (resourceType) {
    case 'scorer':
      return 'Scorers';
    default:
      return resourceType;
  }
};

export const EndpointBindingsDrawer = ({ open, endpointName, bindings, onClose }: EndpointBindingsDrawerProps) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const intl = useIntl();

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

  const expandedSections = useMemo(() => Array.from(bindingsByType.keys()), [bindingsByType]);

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
    }
  };

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content
        componentId="mlflow.gateway.endpoint-bindings.drawer"
        width={480}
        title={
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage
              defaultMessage="Used by ({count})"
              description="Gateway > Endpoint bindings drawer > Title"
              values={{ count: bindings.length }}
            />
          </Typography.Title>
        }
      >
        <Spacer size="md" />
        {bindings.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No resources connected to this endpoint"
                description="Gateway > Endpoint bindings drawer > Empty state"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Resources using endpoint: {name}"
                description="Gateway > Endpoint bindings drawer > Subtitle"
                values={{ name: endpointName }}
              />
            </Typography.Text>

            <Accordion
              componentId="mlflow.gateway.endpoint-bindings.accordion"
              activeKey={expandedSections}
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
                    <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
                      {formatResourceTypePlural(resourceType)} ({typeBindings.length})
                    </span>
                  }
                >
                  <div
                    css={{
                      maxHeight: 8 * 52,
                      overflowY: 'auto',
                    }}
                  >
                    {typeBindings.map((binding) => (
                      <div
                        key={`${binding.endpoint_id}-${binding.resource_type}-${binding.resource_id}`}
                        css={{
                          padding: theme.spacing.sm,
                          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                          '&:last-child': { borderBottom: 'none' },
                        }}
                      >
                        <Typography.Text>{binding.display_name || binding.resource_id}</Typography.Text>
                        <div css={{ marginTop: theme.spacing.xs / 2 }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                            <FormattedMessage
                              defaultMessage="Created {date}"
                              description="Gateway > Endpoint bindings drawer > Created date"
                              values={{
                                date: intl.formatDate(new Date(binding.created_at), {
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric',
                                }),
                              }}
                            />
                          </Typography.Text>
                        </div>
                      </div>
                    ))}
                  </div>
                </Accordion.Panel>
              ))}
            </Accordion>
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};

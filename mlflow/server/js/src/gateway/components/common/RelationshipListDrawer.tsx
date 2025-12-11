import { useCallback, useMemo, useState, type ReactNode } from 'react';
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
import { FormattedMessage } from 'react-intl';

export interface RelationshipSection<T> {
  key: string;
  title: ReactNode;
  items: T[];
  renderItem: (item: T, index: number) => ReactNode;
}

interface RelationshipListDrawerProps<T> {
  open: boolean;
  onClose: () => void;
  componentId: string;
  title: ReactNode;
  subtitle?: ReactNode;
  emptyMessage: ReactNode;
  sections: RelationshipSection<T>[];
  width?: number;
}

export function RelationshipListDrawer<T>({
  open,
  onClose,
  componentId,
  title,
  subtitle,
  emptyMessage,
  sections,
  width = 480,
}: RelationshipListDrawerProps<T>) {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();
  const [expandedSections, setExpandedSections] = useState<string[]>(() => sections.map((s) => s.key));

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      onClose();
      setExpandedSections(sections.map((s) => s.key));
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

  const handleAccordionChange = (keys: string | string[]) => {
    setExpandedSections(Array.isArray(keys) ? keys : [keys]);
  };

  const totalItems = sections.reduce((sum, section) => sum + section.items.length, 0);
  const isEmpty = totalItems === 0;

  return (
    <Drawer.Root modal open={open} onOpenChange={handleOpenChange}>
      <Drawer.Content componentId={componentId} width={width} title={title}>
        <Spacer size="md" />
        {isEmpty ? (
          <Empty description={emptyMessage} />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {subtitle && <Typography.Text color="secondary">{subtitle}</Typography.Text>}

            <Accordion
              componentId={`${componentId}.accordion`}
              activeKey={expandedSections}
              onChange={handleAccordionChange}
              dangerouslyAppendEmotionCSS={accordionStyles}
              dangerouslySetAntdProps={{
                expandIconPosition: 'left',
                expandIcon: getExpandIcon,
              }}
            >
              {sections.map((section) => (
                <Accordion.Panel
                  key={section.key}
                  header={<span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>{section.title}</span>}
                >
                  <div
                    css={{
                      maxHeight: 8 * 64,
                      overflowY: 'auto',
                    }}
                  >
                    {section.items.map((item, index) => (
                      <div
                        key={index}
                        css={{
                          display: 'flex',
                          flexDirection: 'column',
                          gap: theme.spacing.xs,
                          padding: theme.spacing.sm,
                          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                          '&:last-child': { borderBottom: 'none' },
                        }}
                      >
                        {section.renderItem(item, index)}
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
}

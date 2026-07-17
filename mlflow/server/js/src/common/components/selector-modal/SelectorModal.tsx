import { useState, useMemo, useCallback, useEffect } from 'react';
import { Input, Modal, SearchIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { SelectorModalProps, SelectorItem } from './types';

export function SelectorModal<T = string>({
  componentId,
  open,
  onClose,
  onSelect,
  title,
  items,
  searchPlaceholder = 'Search...',
  emptyMessage,
  filterItem,
  renderItem,
  hoverStyle = 'tertiary',
}: SelectorModalProps<T>) {
  const { theme } = useDesignSystemTheme();
  const getHoverBackgroundColor = () => {
    switch (hoverStyle) {
      case 'primary':
        return theme.colors.actionPrimaryBackgroundHover;
      case 'default':
        return theme.colors.actionDefaultBackgroundHover;
      case 'tableSelection':
        return theme.colors.tableBackgroundSelectedHover;
      case 'tertiary':
      default:
        return theme.colors.actionTertiaryBackgroundHover;
    }
  };
  const hoverBackgroundColor = getHoverBackgroundColor();
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    if (!open) {
      setSearchQuery('');
    }
  }, [open]);

  const defaultFilterItem = useCallback((item: SelectorItem<T>, query: string): boolean => {
    const lowerQuery = query.toLowerCase();
    return (
      item.label.toLowerCase().includes(lowerQuery) || (item.description?.toLowerCase().includes(lowerQuery) ?? false)
    );
  }, []);

  const effectiveFilterItem = filterItem ?? defaultFilterItem;

  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) {
      return items;
    }
    return items.filter((item) => effectiveFilterItem(item, searchQuery));
  }, [items, searchQuery, effectiveFilterItem]);

  const handleSelect = useCallback(
    (item: SelectorItem<T>) => {
      onSelect(item.value);
      onClose();
    },
    [onSelect, onClose],
  );

  const handleClose = useCallback(() => {
    setSearchQuery('');
    onClose();
  }, [onClose]);

  return (
    <Modal
      componentId={`${componentId}.modal`}
      title={title}
      visible={open}
      onCancel={handleClose}
      footer={null}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Input
          componentId={`${componentId}.search`}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder={searchPlaceholder}
          prefix={<SearchIcon css={{ color: theme.colors.textSecondary }} />}
          allowClear
        />

        <div
          css={{
            maxHeight: 400,
            overflowY: 'auto',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          {filteredItems.length === 0 ? (
            <div
              css={{
                padding: theme.spacing.lg,
                textAlign: 'center',
                color: theme.colors.textSecondary,
              }}
            >
              {emptyMessage ?? (
                <FormattedMessage
                  defaultMessage="No items found"
                  description="Message shown when no items match the search"
                />
              )}
            </div>
          ) : (
            filteredItems.map((item) => {
              const defaultContent = (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    {item.icon}
                    <Typography.Text bold>{item.label}</Typography.Text>
                  </div>
                  {item.description && (
                    <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                      {item.description}
                    </Typography.Text>
                  )}
                </div>
              );

              return (
                <div
                  key={item.key}
                  data-testid={`${componentId}.item.${item.key}`}
                  onClick={() => handleSelect(item)}
                  css={{
                    padding: theme.spacing.md,
                    cursor: 'pointer',
                    borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                    '&:last-child': {
                      borderBottom: 'none',
                    },
                    '&:hover': {
                      backgroundColor: hoverBackgroundColor,
                    },
                  }}
                >
                  {renderItem ? renderItem(item, defaultContent) : defaultContent}
                </div>
              );
            })
          )}
        </div>
      </div>
    </Modal>
  );
}

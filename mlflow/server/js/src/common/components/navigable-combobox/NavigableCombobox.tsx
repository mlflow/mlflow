import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  TypeaheadComboboxRoot,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  useComboboxState,
  useDesignSystemTheme,
  FormUI,
  ChevronLeftIcon,
  ChevronRightIcon,
  Typography,
} from '@databricks/design-system';
import { SelectorModal } from '../selector-modal/SelectorModal';
import type { SelectorItem } from '../selector-modal/types';
import type {
  NavigableComboboxProps,
  ComboboxMenuItem,
  ComboboxSelectableItem,
  ComboboxGroupItem,
  ComboboxNavigationItem,
  ComboboxSectionHeader,
  ComboboxModalTriggerItem,
  ComboboxView,
} from './types';

type BackNavItem = { type: 'back-nav'; key: string; label: string; targetViewId: string };
type InternalMenuItem<T> = ComboboxMenuItem<T> | BackNavItem;

function isSelectableItem<T>(item: ComboboxMenuItem<T>): item is ComboboxSelectableItem<T> {
  return item.type === 'item';
}

function isGroupItem<T>(item: ComboboxMenuItem<T>): item is ComboboxGroupItem<T> {
  return item.type === 'group';
}

function isNavigationItem<T>(item: ComboboxMenuItem<T>): item is ComboboxNavigationItem {
  return item.type === 'navigation';
}

function isSectionHeader<T>(item: ComboboxMenuItem<T>): item is ComboboxSectionHeader {
  return item.type === 'section';
}

function isModalTriggerItem<T>(item: ComboboxMenuItem<T>): item is ComboboxModalTriggerItem<T> {
  return item.type === 'modal-trigger';
}

function isBackNavItem<T>(item: InternalMenuItem<T>): item is BackNavItem {
  return item.type === 'back-nav';
}

function getAllSelectableItems<T>(views: ComboboxView<T>[]): ComboboxSelectableItem<T>[] {
  const items: ComboboxSelectableItem<T>[] = [];
  for (const view of views) {
    for (const item of view.items) {
      if (isSelectableItem(item)) {
        items.push(item);
      } else if (isGroupItem(item)) {
        items.push(...item.children);
      }
    }
  }
  return items;
}

export function NavigableCombobox<T = string>({
  componentId,
  config,
  value,
  onChange,
  placeholder = 'Search...',
  disabled,
  error,
  minMenuWidth = 300,
  clearOnOpen = true,
  showToggleButton = true,
  allowClear = false,
  valueToDisplayString,
  filterItem,
  renderItem,
  renderModalItem,
  validationState,
  modalHoverStyle = 'tertiary',
}: NavigableComboboxProps<T>) {
  const { theme } = useDesignSystemTheme();
  const [currentViewId, setCurrentViewId] = useState(config.initialViewId);
  const [searchQuery, setSearchQuery] = useState('');
  const [dynamicViews, setDynamicViews] = useState<Map<string, ComboboxView<T>>>(new Map());
  const [modalOpen, setModalOpen] = useState(false);
  const [activeModalTrigger, setActiveModalTrigger] = useState<ComboboxModalTriggerItem<T> | null>(null);
  const setInputValueRef = useRef<((value: string) => void) | null>(null);

  const allViews = useMemo(() => {
    const viewMap = new Map<string, ComboboxView<T>>();
    for (const view of config.views) {
      viewMap.set(view.id, view);
    }
    for (const [id, view] of dynamicViews) {
      viewMap.set(id, view);
    }
    return viewMap;
  }, [config.views, dynamicViews]);

  const currentView = useMemo(
    () => allViews.get(currentViewId) ?? config.views[0],
    [allViews, currentViewId, config.views],
  );

  const allSelectableItems = useMemo(() => getAllSelectableItems(config.views), [config.views]);

  const defaultFilterItem = useCallback((item: ComboboxSelectableItem<T>, query: string): boolean => {
    return item.label.toLowerCase().includes(query.toLowerCase());
  }, []);

  const effectiveFilterItem = filterItem ?? defaultFilterItem;

  const addModalItemsIfMatching = useCallback(
    (modalItems: SelectorItem<T>[], addIfMatches: (item: ComboboxSelectableItem<T>) => void) => {
      for (const modalItem of modalItems) {
        const selectableItem: ComboboxSelectableItem<T> = {
          type: 'item',
          key: modalItem.key,
          label: modalItem.label,
          value: modalItem.value,
          icon: modalItem.icon,
          metadata: modalItem.metadata,
        };
        addIfMatches(selectableItem);
      }
    },
    [],
  );

  const displayItems = useMemo((): InternalMenuItem<T>[] => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const seen = new Set<string>();
      const matchingItems: ComboboxSelectableItem<T>[] = [];

      const addIfMatches = (item: ComboboxSelectableItem<T>) => {
        if (effectiveFilterItem(item, query) && !seen.has(item.key)) {
          seen.add(item.key);
          matchingItems.push(item);
        }
      };

      for (const view of config.views) {
        for (const item of view.items) {
          if (isSelectableItem(item)) {
            addIfMatches(item);
          } else if (isGroupItem(item)) {
            item.children.forEach(addIfMatches);
          } else if (isModalTriggerItem(item)) {
            addModalItemsIfMatching(item.modalItems, addIfMatches);
          }
        }
      }

      return matchingItems;
    }

    const items: InternalMenuItem<T>[] = [];

    if (currentView.backNavigation) {
      items.push({
        type: 'back-nav',
        key: `back-nav-${currentView.backNavigation.targetViewId}`,
        label: currentView.backNavigation.label,
        targetViewId: currentView.backNavigation.targetViewId,
      });
    }

    items.push(...currentView.items);

    return items;
  }, [searchQuery, currentView, config.views, effectiveFilterItem, addModalItemsIfMatching]);

  const selectedDisplayName = useMemo(() => {
    if (value === null || value === undefined) return '';
    if (valueToDisplayString) return valueToDisplayString(value);

    const item = allSelectableItems.find((i) => i.value === value);
    if (item) return item.label;

    for (const view of config.views) {
      for (const viewItem of view.items) {
        if (isModalTriggerItem(viewItem)) {
          const modalItem = viewItem.modalItems.find((mi) => mi.value === value);
          if (modalItem) return modalItem.label;
        }
      }
    }

    return String(value);
  }, [value, valueToDisplayString, allSelectableItems, config.views]);

  const allInternalItems = useMemo((): InternalMenuItem<T>[] => {
    const items: InternalMenuItem<T>[] = [];
    for (const view of allViews.values()) {
      if (view.backNavigation) {
        items.push({
          type: 'back-nav',
          key: `back-nav-${view.backNavigation.targetViewId}`,
          label: view.backNavigation.label,
          targetViewId: view.backNavigation.targetViewId,
        });
      }
      items.push(...view.items);
    }
    return items;
  }, [allViews]);

  const handleInputValueChange = useCallback((val: React.SetStateAction<string>) => {
    if (typeof val === 'string') {
      setSearchQuery(val);
    }
  }, []);

  const handleMenuSelection = useCallback(
    (item: InternalMenuItem<T> | null) => {
      if (!item) {
        onChange(null);
        setSearchQuery('');
        setCurrentViewId(config.initialViewId);
        return;
      }

      if (
        isBackNavItem(item) ||
        isNavigationItem(item as ComboboxMenuItem<T>) ||
        isGroupItem(item as ComboboxMenuItem<T>) ||
        isSectionHeader(item as ComboboxMenuItem<T>) ||
        isModalTriggerItem(item as ComboboxMenuItem<T>)
      ) {
        return;
      }

      if (isSelectableItem(item as ComboboxMenuItem<T>)) {
        onChange((item as ComboboxSelectableItem<T>).value);
        setSearchQuery('');
        setCurrentViewId(config.initialViewId);
      }
    },
    [onChange, config.initialViewId],
  );

  const handleIsOpenChange = useCallback(
    (changes: { isOpen?: boolean }) => {
      if (changes.isOpen && clearOnOpen) {
        setSearchQuery('');
        setCurrentViewId(config.initialViewId);
        setInputValueRef.current?.('');
      }
    },
    [clearOnOpen, config.initialViewId],
  );

  const comboboxState = useComboboxState<InternalMenuItem<T> | null>({
    componentId,
    allItems: allInternalItems,
    items: displayItems,
    setItems: useCallback(() => {}, []),
    setInputValue: handleInputValueChange,
    multiSelect: false,
    itemToString: (item) => {
      if (!item) return '';
      if (isBackNavItem(item)) return item.label;
      return (item as ComboboxMenuItem<T>).label ?? '';
    },
    matcher: useCallback(() => true, []),
    formValue: null,
    formOnChange: handleMenuSelection,
    initialInputValue: selectedDisplayName,
    onIsOpenChange: handleIsOpenChange,
  });

  setInputValueRef.current = comboboxState.setInputValue;

  const isOpen = comboboxState.isOpen;
  useEffect(() => {
    if (!isOpen) {
      setInputValueRef.current?.(selectedDisplayName);
    }
  }, [selectedDisplayName, isOpen]);

  const handleNavigate = useCallback((targetViewId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setCurrentViewId(targetViewId);
    setSearchQuery('');
  }, []);

  const handleGroupClick = useCallback(
    (group: ComboboxGroupItem<T>, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const groupViewId = `group-${group.key}`;

      if (!allViews.has(groupViewId)) {
        const groupView: ComboboxView<T> = {
          id: groupViewId,
          items: group.children,
          backNavigation: {
            label: group.backLabel ?? 'Back',
            targetViewId: currentViewId,
          },
        };
        setDynamicViews((prev) => new Map(prev).set(groupViewId, groupView));
      }

      setCurrentViewId(groupViewId);
      setSearchQuery('');
    },
    [currentViewId, allViews],
  );

  const handleModalTriggerClick = useCallback((trigger: ComboboxModalTriggerItem<T>, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setActiveModalTrigger(trigger);
    setModalOpen(true);
  }, []);

  const handleModalSelect = useCallback(
    (selectedValue: T) => {
      onChange(selectedValue);
      setModalOpen(false);
      setActiveModalTrigger(null);
      setSearchQuery('');
      setCurrentViewId(config.initialViewId);
    },
    [onChange, config.initialViewId],
  );

  const handleModalClose = useCallback(() => {
    setModalOpen(false);
    setActiveModalTrigger(null);
  }, []);

  const renderMenuItem = (item: InternalMenuItem<T>, index: number) => {
    if (isBackNavItem(item)) {
      return (
        <TypeaheadComboboxMenuItem
          key={item.key}
          item={item}
          index={index}
          comboboxState={comboboxState}
          data-testid={`${componentId}.option.${item.key}`}
          onMouseDown={(e) => handleNavigate(item.targetViewId, e)}
        >
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              color: theme.colors.textSecondary,
              borderBottom: `1px solid ${theme.colors.borderDecorative}`,
              paddingBottom: theme.spacing.xs,
              marginBottom: theme.spacing.xs,
            }}
          >
            <ChevronLeftIcon css={{ marginRight: theme.spacing.sm }} />
            <span>{item.label}</span>
          </div>
        </TypeaheadComboboxMenuItem>
      );
    }

    const menuItem = item as ComboboxMenuItem<T>;

    if (isSectionHeader(menuItem)) {
      return (
        <div
          key={menuItem.key}
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            borderBottom: `1px solid ${theme.colors.borderDecorative}`,
            marginBottom: theme.spacing.xs,
          }}
        >
          <Typography.Text bold color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            {menuItem.label}
          </Typography.Text>
        </div>
      );
    }

    if (isNavigationItem(menuItem)) {
      const defaultContent = (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: menuItem.direction === 'forward' ? 'space-between' : 'flex-start',
            width: '100%',
            color: theme.colors.textSecondary,
            ...(menuItem.direction === 'back' && {
              borderBottom: `1px solid ${theme.colors.borderDecorative}`,
              paddingBottom: theme.spacing.xs,
              marginBottom: theme.spacing.xs,
            }),
          }}
        >
          {menuItem.direction === 'back' &&
            (menuItem.icon ?? <ChevronLeftIcon css={{ marginRight: theme.spacing.sm }} />)}
          <span>{menuItem.label}</span>
          {menuItem.direction === 'forward' &&
            (menuItem.icon ?? <ChevronRightIcon css={{ marginLeft: theme.spacing.sm }} />)}
        </div>
      );

      return (
        <TypeaheadComboboxMenuItem
          key={menuItem.key}
          item={item}
          index={index}
          comboboxState={comboboxState}
          data-testid={`${componentId}.option.${menuItem.key}`}
          onMouseDown={(e) => handleNavigate(menuItem.targetViewId, e)}
        >
          {renderItem ? renderItem(menuItem, defaultContent) : defaultContent}
        </TypeaheadComboboxMenuItem>
      );
    }

    if (isGroupItem(menuItem)) {
      const defaultContent = (
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            {menuItem.icon}
            <span>{menuItem.label}</span>
          </div>
          <ChevronRightIcon css={{ color: theme.colors.textSecondary }} />
        </div>
      );

      return (
        <TypeaheadComboboxMenuItem
          key={menuItem.key}
          item={item}
          index={index}
          comboboxState={comboboxState}
          data-testid={`${componentId}.option.${menuItem.key}`}
          onMouseDown={(e) => handleGroupClick(menuItem, e)}
        >
          {renderItem ? renderItem(menuItem, defaultContent) : defaultContent}
        </TypeaheadComboboxMenuItem>
      );
    }

    if (isModalTriggerItem(menuItem)) {
      const defaultContent = (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            width: '100%',
            color: theme.colors.actionPrimaryBackgroundDefault,
          }}
        >
          {menuItem.icon}
          <span>{menuItem.label}</span>
        </div>
      );

      return (
        <TypeaheadComboboxMenuItem
          key={menuItem.key}
          item={item}
          index={index}
          comboboxState={comboboxState}
          data-testid={`${componentId}.option.${menuItem.key}`}
          onMouseDown={(e) => handleModalTriggerClick(menuItem, e)}
        >
          {renderItem ? renderItem(menuItem, defaultContent) : defaultContent}
        </TypeaheadComboboxMenuItem>
      );
    }

    const selectableItem = menuItem as ComboboxSelectableItem<T>;
    const defaultContent = (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        {selectableItem.icon}
        <span>{selectableItem.label}</span>
      </div>
    );

    return (
      <TypeaheadComboboxMenuItem
        key={selectableItem.key}
        item={item}
        index={index}
        comboboxState={comboboxState}
        data-testid={`${componentId}.option.${selectableItem.key}`}
      >
        {renderItem ? renderItem(selectableItem, defaultContent) : defaultContent}
      </TypeaheadComboboxMenuItem>
    );
  };

  return (
    <>
      <TypeaheadComboboxRoot id={componentId} comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          placeholder={placeholder}
          comboboxState={comboboxState}
          validationState={error ? 'error' : validationState}
          disabled={disabled}
          showComboboxToggleButton={showToggleButton}
          allowClear={allowClear}
        />
        <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth minWidth={minMenuWidth}>
          {displayItems.map((item, index) => renderMenuItem(item, index))}
        </TypeaheadComboboxMenu>
      </TypeaheadComboboxRoot>
      {error && <FormUI.Message type="error" message={error} />}

      {activeModalTrigger && (
        <SelectorModal<T>
          componentId={`${componentId}.modal`}
          open={modalOpen}
          onClose={handleModalClose}
          onSelect={handleModalSelect}
          title={activeModalTrigger.modalTitle}
          items={activeModalTrigger.modalItems}
          searchPlaceholder={activeModalTrigger.modalSearchPlaceholder}
          emptyMessage={activeModalTrigger.modalEmptyMessage}
          renderItem={renderModalItem}
          hoverStyle={activeModalTrigger.modalHoverStyle ?? modalHoverStyle}
        />
      )}
    </>
  );
}

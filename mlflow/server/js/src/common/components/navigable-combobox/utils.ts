import type {
  ComboboxSelectableItem,
  ComboboxGroupItem,
  ComboboxNavigationItem,
  ComboboxSectionHeader,
  ComboboxView,
  NavigableComboboxConfig,
} from './types';

export function createItem<T = string>(
  key: string,
  label: string,
  value: T,
  options?: { icon?: React.ReactNode; metadata?: Record<string, unknown> },
): ComboboxSelectableItem<T> {
  return { type: 'item', key, label, value, ...options };
}

export function createGroup<T = string>(
  key: string,
  label: string,
  children: ComboboxSelectableItem<T>[],
  options?: { icon?: React.ReactNode; backLabel?: string },
): ComboboxGroupItem<T> {
  return { type: 'group', key, label, children, ...options };
}

export function createForwardNav(
  key: string,
  label: string,
  targetViewId: string,
  icon?: React.ReactNode,
): ComboboxNavigationItem {
  return { type: 'navigation', key, label, targetViewId, direction: 'forward', icon };
}

export function createBackNav(
  key: string,
  label: string,
  targetViewId: string,
  icon?: React.ReactNode,
): ComboboxNavigationItem {
  return { type: 'navigation', key, label, targetViewId, direction: 'back', icon };
}

export function createSection(key: string, label: string): ComboboxSectionHeader {
  return { type: 'section', key, label };
}

export function createView<T = string>(
  id: string,
  items: (ComboboxSelectableItem<T> | ComboboxGroupItem<T> | ComboboxNavigationItem | ComboboxSectionHeader)[],
  backNavigation?: { label: string; targetViewId: string },
): ComboboxView<T> {
  return { id, items, backNavigation };
}

export function createConfig<T = string>(views: ComboboxView<T>[], initialViewId: string): NavigableComboboxConfig<T> {
  return { views, initialViewId };
}

export function createTwoTierConfig<T = string>(options: {
  mainViewId: string;
  mainItems: (ComboboxSelectableItem<T> | ComboboxGroupItem<T>)[];
  moreViewId: string;
  moreViewLabel: string;
  moreItems: ComboboxSelectableItem<T>[];
  backLabel?: string;
}): NavigableComboboxConfig<T> {
  const { mainViewId, mainItems, moreViewId, moreViewLabel, moreItems, backLabel = 'Back' } = options;

  const mainView: ComboboxView<T> = {
    id: mainViewId,
    items: [
      ...mainItems,
      ...(moreItems.length > 0 ? [createForwardNav('nav-to-more', moreViewLabel, moreViewId)] : []),
    ],
  };

  const moreView: ComboboxView<T> = {
    id: moreViewId,
    items: moreItems,
    backNavigation: { label: backLabel, targetViewId: mainViewId },
  };

  return {
    views: moreItems.length > 0 ? [mainView, moreView] : [mainView],
    initialViewId: mainViewId,
  };
}

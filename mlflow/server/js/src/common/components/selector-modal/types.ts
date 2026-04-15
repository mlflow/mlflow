import type { ReactNode } from 'react';

export interface SelectorItem<T = string> {
  key: string;
  label: string;
  value: T;
  description?: string;
  icon?: ReactNode;
  metadata?: Record<string, unknown>;
}

export interface SelectorModalProps<T = string> {
  componentId: string;
  open: boolean;
  onClose: () => void;
  onSelect: (value: T) => void;
  title: string;
  items: SelectorItem<T>[];
  searchPlaceholder?: string;
  emptyMessage?: string;
  filterItem?: (item: SelectorItem<T>, query: string) => boolean;
  renderItem?: (item: SelectorItem<T>, defaultContent: ReactNode) => ReactNode;
  hoverStyle?: 'primary' | 'tertiary' | 'default' | 'tableSelection';
}

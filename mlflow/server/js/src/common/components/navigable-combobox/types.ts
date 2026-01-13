import type { SelectorItem } from '../selector-modal/types';

export interface ComboboxSelectableItem<T = string> {
  type: 'item';
  key: string;
  label: string;
  value: T;
  icon?: React.ReactNode;
  metadata?: Record<string, unknown>;
}

export interface ComboboxGroupItem<T = string> {
  type: 'group';
  key: string;
  label: string;
  children: ComboboxSelectableItem<T>[];
  icon?: React.ReactNode;
  backLabel?: string;
}

export interface ComboboxNavigationItem {
  type: 'navigation';
  key: string;
  label: string;
  targetViewId: string;
  direction: 'forward' | 'back';
  icon?: React.ReactNode;
}

export interface ComboboxSectionHeader {
  type: 'section';
  key: string;
  label: string;
}

export interface ComboboxModalTriggerItem<T = string> {
  type: 'modal-trigger';
  key: string;
  label: string;
  icon?: React.ReactNode;
  modalTitle: string;
  modalSearchPlaceholder?: string;
  modalEmptyMessage?: string;
  modalItems: SelectorItem<T>[];
  modalHoverStyle?: 'primary' | 'tertiary' | 'default' | 'tableSelection';
}

export type ComboboxMenuItem<T = string> =
  | ComboboxSelectableItem<T>
  | ComboboxGroupItem<T>
  | ComboboxNavigationItem
  | ComboboxSectionHeader
  | ComboboxModalTriggerItem<T>;

export interface ComboboxView<T = string> {
  id: string;
  items: ComboboxMenuItem<T>[];
  backNavigation?: {
    label: string;
    targetViewId: string;
  };
}

export interface NavigableComboboxConfig<T = string> {
  views: ComboboxView<T>[];
  initialViewId: string;
}

export interface NavigableComboboxProps<T = string> {
  componentId: string;
  config: NavigableComboboxConfig<T>;
  value: T | null;
  onChange: (value: T | null) => void;
  placeholder?: string;
  disabled?: boolean;
  error?: string;
  minMenuWidth?: number;
  clearOnOpen?: boolean;
  showToggleButton?: boolean;
  allowClear?: boolean;
  valueToDisplayString?: (value: T) => string;
  filterItem?: (item: ComboboxSelectableItem<T>, query: string) => boolean;
  renderItem?: (item: ComboboxMenuItem<T>, defaultRender: React.ReactNode) => React.ReactNode;
  renderModalItem?: (item: SelectorItem<T>, defaultRender: React.ReactNode) => React.ReactNode;
  validationState?: 'error' | 'warning' | 'success';
  modalHoverStyle?: 'primary' | 'tertiary' | 'default' | 'tableSelection';
}

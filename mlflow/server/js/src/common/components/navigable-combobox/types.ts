export interface ComboboxSelectableItem<T = string> {
  type: 'item';
  key: string;
  label: string;
  value: T;
  /** Optional icon to display before the label */
  icon?: React.ReactNode;
  /** Optional metadata for custom rendering or filtering */
  metadata?: Record<string, unknown>;
}

export interface ComboboxGroupItem<T = string> {
  type: 'group';
  key: string;
  label: string;
  /** Items shown when navigating into this group */
  children: ComboboxSelectableItem<T>[];
  /** Optional icon to display before the label */
  icon?: React.ReactNode;
  /** Custom back navigation label when inside this group */
  backLabel?: string;
}

export interface ComboboxNavigationItem {
  type: 'navigation';
  key: string;
  label: string;
  targetViewId: string;
  direction: 'forward' | 'back';
  /** Optional icon to display (overrides default chevron) */
  icon?: React.ReactNode;
}

export interface ComboboxSectionHeader {
  type: 'section';
  key: string;
  label: string;
}

export type ComboboxMenuItem<T = string> =
  | ComboboxSelectableItem<T>
  | ComboboxGroupItem<T>
  | ComboboxNavigationItem
  | ComboboxSectionHeader;

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
  /** Clear input and reset to initial view when dropdown opens */
  clearOnOpen?: boolean;
  showToggleButton?: boolean;
  allowClear?: boolean;
  /** Convert value to display string when closed */
  valueToDisplayString?: (value: T) => string;
  /** Custom filter for search results */
  filterItem?: (item: ComboboxSelectableItem<T>, query: string) => boolean;
  /** Custom renderer for menu items */
  renderItem?: (item: ComboboxMenuItem<T>, defaultRender: React.ReactNode) => React.ReactNode;
  validationState?: 'error' | 'warning' | 'success';
}

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { useIntl } from 'react-intl';

export interface UseItemSelectionResult {
  /** Items currently selected for display (subset of allItems) */
  displayedItems: string[];
  /** Whether all items are selected */
  isAllSelected: boolean;
  /** Label text for the selector dropdown */
  selectorLabel: string;
  /** Toggle all items on/off */
  handleSelectAllToggle: () => void;
  /** Toggle a single item on/off */
  handleItemToggle: (itemName: string) => void;
}

/**
 * Reusable hook for managing item selection state in chart components.
 *
 * @param allItems - The full sorted list of available item names
 * @param labels - Label strings for the selector (allSelected and noneSelected)
 * @param resetKey - When this value changes, selection resets to all selected
 * @returns Selection state and handlers
 */
export function useItemSelection(
  allItems: string[],
  labels: { allSelected: string; noneSelected: string },
  resetKey?: unknown,
): UseItemSelectionResult {
  const intl = useIntl();

  // null means all selected (default), array means specific selection
  const [selectedItems, setSelectedItems] = useState<string[] | null>(null);

  // Reset selection when resetKey changes
  const prevResetKey = useRef(resetKey);
  useEffect(() => {
    if (prevResetKey.current !== resetKey) {
      prevResetKey.current = resetKey;
      setSelectedItems(null);
    }
  }, [resetKey]);

  const isAllSelected = selectedItems === null;
  const displayedItems = isAllSelected ? allItems : selectedItems.filter((item) => allItems.includes(item));

  const handleSelectAllToggle = useCallback(() => {
    setSelectedItems(isAllSelected ? [] : null);
  }, [isAllSelected]);

  const handleItemToggle = useCallback(
    (itemName: string) => {
      setSelectedItems((prev) => {
        if (prev === null) {
          return allItems.filter((m) => m !== itemName);
        }
        const newSelection = prev.includes(itemName) ? prev.filter((m) => m !== itemName) : [...prev, itemName];
        return newSelection.length === allItems.length ? null : newSelection;
      });
    },
    [allItems],
  );

  const selectorLabel = useMemo(() => {
    if (isAllSelected) {
      return labels.allSelected;
    }
    if (displayedItems.length === 0) {
      return labels.noneSelected;
    }
    return intl.formatMessage(
      {
        defaultMessage: '{count} selected',
        description: 'Label for item selector showing count of selected items',
      },
      { count: displayedItems.length },
    );
  }, [isAllSelected, displayedItems, intl, labels]);

  return {
    displayedItems,
    isAllSelected,
    selectorLabel,
    handleSelectAllToggle,
    handleItemToggle,
  };
}

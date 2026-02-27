import { useState, useCallback, useMemo } from 'react';
import { useIntl } from 'react-intl';

export interface UseToolSelectionResult {
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
 * Reusable hook for managing tool selection state in chart components.
 * Follows the same pattern as TraceCostOverTimeChart's item selector.
 *
 * @param allItems - The full sorted list of available tool names
 * @returns Selection state and handlers
 */
export function useToolSelection(allItems: string[]): UseToolSelectionResult {
  const intl = useIntl();

  // null means all selected (default), array means specific selection
  const [selectedItems, setSelectedItems] = useState<string[] | null>(null);

  const isAllSelected = selectedItems === null;
  const displayedItems = isAllSelected ? allItems : selectedItems.filter((item) => allItems.includes(item));

  const handleSelectAllToggle = useCallback(() => {
    setSelectedItems((prev) => (prev === null ? [] : null));
  }, []);

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
      return intl.formatMessage({
        defaultMessage: 'All tools',
        description: 'Label for tool selector when all tools are selected',
      });
    }
    if (displayedItems.length === 0) {
      return intl.formatMessage({
        defaultMessage: 'No tools selected',
        description: 'Label for tool selector when no tools are selected',
      });
    }
    return intl.formatMessage(
      {
        defaultMessage: '{count} selected',
        description: 'Label for tool selector showing count of selected tools',
      },
      { count: displayedItems.length },
    );
  }, [isAllSelected, displayedItems, intl]);

  return {
    displayedItems,
    isAllSelected,
    selectorLabel,
    handleSelectAllToggle,
    handleItemToggle,
  };
}

import React from 'react';
import {
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

interface ItemSelectorProps {
  /** Component ID for telemetry */
  componentId: string;
  /** All available item names */
  itemNames: string[];
  /** Items currently displayed */
  displayedItems: string[];
  /** Whether all items are selected */
  isAllSelected: boolean;
  /** Label text for the selector */
  selectorLabel: string;
  /** Handler to toggle all items */
  onSelectAllToggle: () => void;
  /** Handler to toggle a single item */
  onItemToggle: (itemName: string) => void;
  /** Optional test ID for the trigger element */
  'data-testid'?: string;
}

export const ItemSelector: React.FC<ItemSelectorProps> = ({
  componentId,
  itemNames,
  displayedItems,
  isAllSelected,
  selectorLabel,
  onSelectAllToggle,
  onItemToggle,
  'data-testid': dataTestId,
}) => {
  if (itemNames.length === 0) {
    return null;
  }

  return (
    <DialogCombobox componentId={componentId} label={selectorLabel} multiSelect value={[]}>
      <DialogComboboxTrigger allowClear={false} css={{ minWidth: 120 }} data-testid={dataTestId} />
      <DialogComboboxContent maxHeight={300} align="end">
        <DialogComboboxOptionList>
          <DialogComboboxOptionListSearch>
            <DialogComboboxOptionListCheckboxItem
              key="__select_all__"
              value="__select_all__"
              checked={isAllSelected}
              onChange={onSelectAllToggle}
            >
              <FormattedMessage
                defaultMessage="Select All"
                description="Option to select all items in the item selector"
              />
            </DialogComboboxOptionListCheckboxItem>
            {itemNames.map((itemName) => (
              <DialogComboboxOptionListCheckboxItem
                key={itemName}
                value={itemName}
                checked={isAllSelected || displayedItems.includes(itemName)}
                onChange={() => onItemToggle(itemName)}
              >
                {itemName}
              </DialogComboboxOptionListCheckboxItem>
            ))}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

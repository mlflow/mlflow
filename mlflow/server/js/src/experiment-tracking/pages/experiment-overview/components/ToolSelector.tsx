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

interface ToolSelectorProps {
  /** Component ID for telemetry */
  componentId: string;
  /** All available tool names */
  toolNames: string[];
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
}

export const ToolSelector: React.FC<ToolSelectorProps> = ({
  componentId,
  toolNames,
  displayedItems,
  isAllSelected,
  selectorLabel,
  onSelectAllToggle,
  onItemToggle,
}) => {
  if (toolNames.length === 0) {
    return null;
  }

  return (
    <DialogCombobox componentId={componentId} label={selectorLabel} multiSelect value={[]}>
      <DialogComboboxTrigger allowClear={false} css={{ minWidth: 120 }} />
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
                description="Option to select all tools in the tool selector"
              />
            </DialogComboboxOptionListCheckboxItem>
            {toolNames.map((toolName) => (
              <DialogComboboxOptionListCheckboxItem
                key={toolName}
                value={toolName}
                checked={isAllSelected || displayedItems.includes(toolName)}
                onChange={() => onItemToggle(toolName)}
              >
                {toolName}
              </DialogComboboxOptionListCheckboxItem>
            ))}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

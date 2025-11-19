import {
  ColumnsIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxTrigger,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { SessionTableColumn } from './types';

export const GenAIChatSessionsToolbar = ({
  columns,
  columnVisibility,
  setColumnVisibility,
}: {
  columns: SessionTableColumn[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm }}>
      <DialogCombobox
        componentId="mlflow.chat-sessions.table-column-selector"
        label={
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <ColumnsIcon />
            <FormattedMessage
              defaultMessage="Columns"
              description="Columns label for the chat sessions table column selector"
            />
          </div>
        }
        multiSelect
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {Object.entries(columnVisibility).map(([columnId, isVisible]) => {
              return (
                <DialogComboboxOptionListCheckboxItem
                  key={columnId}
                  value={columnId}
                  onChange={() => {
                    const newColumnVisibility = { ...columnVisibility };
                    newColumnVisibility[columnId] = !isVisible;
                    setColumnVisibility(newColumnVisibility);
                  }}
                  checked={isVisible}
                >
                  {columns.find((column) => column.id === columnId)?.header}
                </DialogComboboxOptionListCheckboxItem>
              );
            })}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
    </div>
  );
};

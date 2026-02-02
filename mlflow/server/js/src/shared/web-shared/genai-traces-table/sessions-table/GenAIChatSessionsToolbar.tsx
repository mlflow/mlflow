import type { RowSelectionState } from '@tanstack/react-table';

import {
  ColumnsIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxTrigger,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { GenAIChatSessionsActions } from './GenAIChatSessionsActions';
import type { SessionTableColumn, SessionTableRow } from './types';
import { GenAiTracesTableSearchInput } from '../GenAiTracesTableSearchInput';
import type { TraceActions } from '../types';

export const GenAIChatSessionsToolbar = ({
  columns,
  columnVisibility,
  setColumnVisibility,
  searchQuery,
  setSearchQuery,
  traceActions,
  experimentId,
  selectedSessions,
  setRowSelection,
  addons,
}: {
  columns: SessionTableColumn[];
  columnVisibility: Record<string, boolean>;
  setColumnVisibility: (columnVisibility: Record<string, boolean>) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  traceActions?: TraceActions;
  experimentId: string;
  selectedSessions: SessionTableRow[];
  setRowSelection?: React.Dispatch<React.SetStateAction<RowSelectionState>>;
  addons?: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm }}>
      <GenAiTracesTableSearchInput
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search chat sessions by input',
          description: 'Placeholder text for the search input in the chat sessions table',
        })}
      />
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
      {traceActions && (
        <GenAIChatSessionsActions
          experimentId={experimentId}
          selectedSessions={selectedSessions}
          traceActions={traceActions}
          setRowSelection={setRowSelection}
        />
      )}
      {addons}
    </div>
  );
};

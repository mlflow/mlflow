import {
  Button,
  ChevronDownIcon,
  ColumnsIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';

export enum ApiKeysColumn {
  KEY_NAME = 'key_name',
  PROVIDER = 'provider',
  ENDPOINTS = 'endpoints',
  USED_BY = 'used_by',
  LAST_UPDATED = 'last_updated',
  CREATED = 'created',
}

const COLUMN_LABELS: Record<ApiKeysColumn, string> = {
  [ApiKeysColumn.KEY_NAME]: 'Key name',
  [ApiKeysColumn.PROVIDER]: 'Provider',
  [ApiKeysColumn.ENDPOINTS]: 'Endpoints',
  [ApiKeysColumn.USED_BY]: 'Used by',
  [ApiKeysColumn.LAST_UPDATED]: 'Last updated',
  [ApiKeysColumn.CREATED]: 'Created',
};

// Columns that cannot be hidden
const REQUIRED_COLUMNS = [ApiKeysColumn.KEY_NAME];

interface ApiKeysColumnsButtonProps {
  visibleColumns: ApiKeysColumn[];
  onColumnsChange: (columns: ApiKeysColumn[]) => void;
}

export const ApiKeysColumnsButton = ({ visibleColumns, onColumnsChange }: ApiKeysColumnsButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const allColumns = Object.values(ApiKeysColumn);
  const toggleableColumns = allColumns.filter((col) => !REQUIRED_COLUMNS.includes(col));

  const handleToggle = (column: ApiKeysColumn) => {
    if (REQUIRED_COLUMNS.includes(column)) return;

    const newColumns = visibleColumns.includes(column)
      ? visibleColumns.filter((c) => c !== column)
      : [...visibleColumns, column];
    onColumnsChange(newColumns);
  };

  return (
    <DialogCombobox componentId="mlflow.gateway.api-keys-list.columns-dropdown" label="Columns" multiSelect>
      <DialogComboboxCustomButtonTriggerWrapper>
        <Button
          componentId="mlflow.gateway.api-keys-list.columns-button"
          endIcon={<ChevronDownIcon />}
          data-testid="api-keys-column-selector-button"
        >
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <ColumnsIcon />
            {intl.formatMessage({
              defaultMessage: 'Columns',
              description: 'Columns button label',
            })}
          </div>
        </Button>
      </DialogComboboxCustomButtonTriggerWrapper>
      <DialogComboboxContent minWidth={200}>
        <DialogComboboxOptionList>
          {toggleableColumns.map((column) => (
            <DialogComboboxOptionListCheckboxItem
              key={column}
              value={column}
              checked={visibleColumns.includes(column)}
              onChange={() => handleToggle(column)}
            >
              {COLUMN_LABELS[column]}
            </DialogComboboxOptionListCheckboxItem>
          ))}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

export const DEFAULT_VISIBLE_COLUMNS: ApiKeysColumn[] = Object.values(ApiKeysColumn);

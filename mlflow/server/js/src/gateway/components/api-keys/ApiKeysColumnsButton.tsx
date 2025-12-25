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

export type ToggleableApiKeysColumn = Exclude<ApiKeysColumn, ApiKeysColumn.KEY_NAME>;

const COLUMN_LABELS: Record<ToggleableApiKeysColumn, string> = {
  [ApiKeysColumn.PROVIDER]: 'Provider',
  [ApiKeysColumn.ENDPOINTS]: 'Endpoints',
  [ApiKeysColumn.USED_BY]: 'Used by',
  [ApiKeysColumn.LAST_UPDATED]: 'Last updated',
  [ApiKeysColumn.CREATED]: 'Created',
};

const TOGGLEABLE_COLUMNS = Object.keys(COLUMN_LABELS) as ToggleableApiKeysColumn[];

interface ApiKeysColumnsButtonProps {
  visibleColumns: ToggleableApiKeysColumn[];
  onColumnsChange: (columns: ToggleableApiKeysColumn[]) => void;
}

export const ApiKeysColumnsButton = ({ visibleColumns, onColumnsChange }: ApiKeysColumnsButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const handleToggle = (column: ToggleableApiKeysColumn) => {
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
          {TOGGLEABLE_COLUMNS.map((column) => (
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

export const DEFAULT_VISIBLE_COLUMNS: ToggleableApiKeysColumn[] = [...TOGGLEABLE_COLUMNS];

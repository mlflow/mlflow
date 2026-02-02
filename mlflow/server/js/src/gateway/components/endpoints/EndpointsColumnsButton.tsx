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

export enum EndpointsColumn {
  NAME = 'name',
  PROVIDER = 'provider',
  MODELS = 'models',
  USED_BY = 'used_by',
  LAST_MODIFIED = 'last_modified',
  CREATED = 'created',
}

const COLUMN_LABELS: Record<EndpointsColumn, string> = {
  [EndpointsColumn.NAME]: 'Name',
  [EndpointsColumn.PROVIDER]: 'Provider',
  [EndpointsColumn.MODELS]: 'Models',
  [EndpointsColumn.USED_BY]: 'Used by',
  [EndpointsColumn.LAST_MODIFIED]: 'Last modified',
  [EndpointsColumn.CREATED]: 'Created',
};

// Columns that cannot be hidden
const REQUIRED_COLUMNS = [EndpointsColumn.NAME];

interface EndpointsColumnsButtonProps {
  visibleColumns: EndpointsColumn[];
  onColumnsChange: (columns: EndpointsColumn[]) => void;
}

export const EndpointsColumnsButton = ({ visibleColumns, onColumnsChange }: EndpointsColumnsButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const allColumns = Object.values(EndpointsColumn);
  const toggleableColumns = allColumns.filter((col) => !REQUIRED_COLUMNS.includes(col));

  const handleToggle = (column: EndpointsColumn) => {
    if (REQUIRED_COLUMNS.includes(column)) return;

    const newColumns = visibleColumns.includes(column)
      ? visibleColumns.filter((c) => c !== column)
      : [...visibleColumns, column];
    onColumnsChange(newColumns);
  };

  return (
    <DialogCombobox componentId="mlflow.gateway.endpoints-list.columns-dropdown" label="Columns" multiSelect>
      <DialogComboboxCustomButtonTriggerWrapper>
        <Button
          componentId="mlflow.gateway.endpoints-list.columns-button"
          endIcon={<ChevronDownIcon />}
          data-testid="endpoints-column-selector-button"
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

export const DEFAULT_VISIBLE_COLUMNS: EndpointsColumn[] = Object.values(EndpointsColumn).filter(
  (col) => col !== EndpointsColumn.CREATED,
);

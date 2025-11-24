import { useCallback, useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  CloseSmallIcon,
  FilterIcon,
  FormUI,
  Input,
  PlusIcon,
  Popover,
  SimpleSelect,
  SimpleSelectOption,
  useDesignSystemTheme,
  XCircleFillIcon,
} from '@databricks/design-system';
import { useIntl, FormattedMessage } from 'react-intl';

interface Filter {
  field: string;
  operator: string;
  value: string;
  key?: string;
}

const OPERATOR_LABELS: Record<string, string> = {
  '=': 'Equals',
  '!=': 'Not Equals',
  '>': 'Greater Than',
  '<': 'Less Than',
  '>=': 'Greater Than or Equals',
  '<=': 'Less Than or Equals',
  'CONTAINS': 'Contains',
};

export const ChartsFilters = () => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [appliedFilters, setAppliedFilters] = useState<Filter[]>([]);
  const [localFilters, setLocalFilters] = useState<Filter[]>([{ field: '', operator: '=', value: '' }]);
  const [isOpen, setIsOpen] = useState(false);

  // Field options
  const fieldOptions = [
    { value: 'tags', label: 'Tags' },
    { value: 'status', label: 'Status' },
    { value: 'source_run', label: 'Source run' },
  ];

  // Operator options
  const operatorOptions = ['=', '!=', '>', '<', '>=', '<=', 'CONTAINS'];

  const hasActiveFilters = appliedFilters.length > 0 && appliedFilters.some(f => f.field && f.value);

  const clearFilters = useCallback(() => {
    setAppliedFilters([]);
    setLocalFilters([{ field: '', operator: '=', value: '' }]);
  }, []);

  const applyFilters = useCallback(() => {
    const validFilters = localFilters.filter(f => f.field && f.value);
    setAppliedFilters(validFilters);
    setIsOpen(false);
  }, [localFilters]);

  const updateFilter = (index: number, updates: Partial<Filter>) => {
    const newFilters = [...localFilters];
    newFilters[index] = { ...newFilters[index], ...updates };
    setLocalFilters(newFilters);
  };

  const addFilter = () => {
    setLocalFilters([...localFilters, { field: '', operator: '=', value: '' }]);
  };

  const removeFilter = (index: number) => {
    const newFilters = [...localFilters];
    newFilters.splice(index, 1);
    if (newFilters.length === 0) {
      newFilters.push({ field: '', operator: '=', value: '' });
    }
    setLocalFilters(newFilters);
  };

  return (
    <Popover.Root componentId="mlflow.experiment-charts.filters" open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <Button
          endIcon={<ChevronDownIcon />}
          componentId="mlflow.experiment-charts.filter-button"
          css={{
            border: hasActiveFilters ? `1px solid ${theme.colors.actionDefaultBorderFocus} !important` : '',
            backgroundColor: hasActiveFilters ? `${theme.colors.actionDefaultBackgroundHover} !important` : '',
          }}
        >
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              alignItems: 'center',
            }}
          >
            <FilterIcon />
            {intl.formatMessage(
              {
                defaultMessage: 'Filters{numFilters}',
                description: 'Filter dropdown button for charts view',
              },
              {
                numFilters: hasActiveFilters ? ` (${appliedFilters.filter(f => f.field && f.value).length})` : '',
              },
            )}
            {hasActiveFilters && (
              <XCircleFillIcon
                css={{
                  fontSize: 12,
                  cursor: 'pointer',
                  color: theme.colors.grey400,
                  '&:hover': {
                    color: theme.colors.grey600,
                  },
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  clearFilters();
                }}
              />
            )}
          </div>
        </Button>
      </Popover.Trigger>
      <Popover.Content align="start" css={{ padding: theme.spacing.md, minWidth: 600 }}>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.lg,
          }}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
            }}
          >
            {localFilters.map((filter, index) => (
              <div
                key={index}
                css={{
                  display: 'flex',
                  flexDirection: 'row',
                  gap: theme.spacing.sm,
                }}
              >
                <div css={{ display: 'flex', flexDirection: 'column' }}>
                  <FormUI.Label htmlFor={`filter-field-${index}`}>
                    <FormattedMessage
                      defaultMessage="Field"
                      description="Label for field selector in filter"
                    />
                  </FormUI.Label>
                  <SimpleSelect
                    aria-label="Field"
                    componentId="mlflow.experiment-charts.filter-field"
                    id={`filter-field-${index}`}
                    placeholder="Select field"
                    width={180}
                    value={filter.field}
                    onChange={(e) => updateFilter(index, { field: e.target.value, value: '' })}
                    contentProps={{
                      style: { zIndex: theme.options.zIndexBase + 100 },
                    }}
                  >
                    {fieldOptions.map((option) => (
                      <SimpleSelectOption key={option.value} value={option.value}>
                        {option.label}
                      </SimpleSelectOption>
                    ))}
                  </SimpleSelect>
                </div>

                {filter.field === 'tags' && (
                  <div css={{ display: 'flex', flexDirection: 'column' }}>
                    <FormUI.Label htmlFor={`filter-key-${index}`}>
                      <FormattedMessage
                        defaultMessage="Key"
                        description="Label for key field for tags in filter"
                      />
                    </FormUI.Label>
                    <Input
                      aria-label="Key"
                      componentId="mlflow.experiment-charts.filter-key"
                      id={`filter-key-${index}`}
                      type="text"
                      css={{ width: 180 }}
                      placeholder="Key"
                      value={filter.key || ''}
                      onChange={(e) => updateFilter(index, { key: e.target.value })}
                    />
                  </div>
                )}

                <div css={{ display: 'flex', flexDirection: 'column' }}>
                  <FormUI.Label htmlFor={`filter-operator-${index}`}>
                    <FormattedMessage
                      defaultMessage="Operator"
                      description="Label for operator selector in filter"
                    />
                  </FormUI.Label>
                  <SimpleSelect
                    aria-label="Operator"
                    componentId="mlflow.experiment-charts.filter-operator"
                    id={`filter-operator-${index}`}
                    placeholder="Select"
                    width={180}
                    value={filter.operator}
                    onChange={(e) => updateFilter(index, { operator: e.target.value })}
                    contentProps={{
                      style: { zIndex: theme.options.zIndexBase + 100 },
                    }}
                  >
                    {operatorOptions.map((op) => (
                      <SimpleSelectOption key={op} value={op}>
                        {OPERATOR_LABELS[op] || op}
                      </SimpleSelectOption>
                    ))}
                  </SimpleSelect>
                </div>

                <div css={{ display: 'flex', flexDirection: 'column' }}>
                  <FormUI.Label htmlFor={`filter-value-${index}`}>
                    <FormattedMessage
                      defaultMessage="Value"
                      description="Label for value input in filter"
                    />
                  </FormUI.Label>
                  <Input
                    aria-label="Value"
                    componentId="mlflow.experiment-charts.filter-value"
                    id={`filter-value-${index}`}
                    type="text"
                    css={{ width: 180 }}
                    placeholder="Value"
                    value={filter.value}
                    onChange={(e) => updateFilter(index, { value: e.target.value })}
                  />
                </div>

                <div css={{ alignSelf: 'flex-end' }}>
                  <Button
                    componentId="mlflow.experiment-charts.filter-delete-button"
                    type="tertiary"
                    icon={<CloseSmallIcon />}
                    onClick={() => removeFilter(index)}
                  />
                </div>
              </div>
            ))}
          </div>

          <div>
            <Button
              componentId="mlflow.experiment-charts.add-filter-button"
              icon={<PlusIcon />}
              onClick={addFilter}
            >
              <FormattedMessage
                defaultMessage="Add filter"
                description="Button to add a new filter"
              />
            </Button>
          </div>

          <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              componentId="mlflow.experiment-charts.apply-filters-button"
              type="primary"
              onClick={applyFilters}
            >
              <FormattedMessage
                defaultMessage="Apply filters"
                description="Button to apply the filters"
              />
            </Button>
          </div>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};

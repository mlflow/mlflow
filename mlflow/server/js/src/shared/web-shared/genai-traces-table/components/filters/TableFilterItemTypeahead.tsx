import { useMemo, useState } from 'react';

import {
  useDesignSystemTheme,
  DialogComboboxOptionListSelectItem,
  DialogComboboxOptionListSearch,
  DialogComboboxContent,
  DialogComboboxTrigger,
  DialogComboboxOptionList,
  DialogCombobox,
} from '@databricks/design-system';

import type { TableFilterOption } from '../../types';

interface SimpleQuery<T> {
  data?: T;
  isLoading?: boolean;
}

export const TableFilterItemTypeahead = ({
  id,
  item,
  options,
  query,
  onChange,
  placeholder,
  width,
  canSearchCustomValue,
}: {
  id: string;
  item?: TableFilterOption;
  options?: TableFilterOption[];
  query?: SimpleQuery<TableFilterOption[]>;
  onChange: (value: string) => void;
  placeholder: string;
  width: number;
  canSearchCustomValue: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  const [searchValue, setSearchValue] = useState<string>('');

  const displayOptions = useMemo(() => query?.data ?? options ?? [], [query?.data, options]);
  const optionValues = useMemo(() => displayOptions.map((option) => option.value), [displayOptions]);

  return (
    <DialogCombobox
      componentId="mlflow.evaluations_review.table_ui.filter_column"
      value={item?.value ? [item.value] : []}
      id={id}
    >
      <DialogComboboxTrigger
        withInlineLabel={false}
        placeholder={placeholder}
        renderDisplayedValue={() => item?.renderValue() ?? ''}
        width={width}
        allowClear={false}
      />
      <DialogComboboxContent
        width={width}
        style={{ zIndex: theme.options.zIndexBase + 100 }}
        loading={query?.isLoading}
      >
        <DialogComboboxOptionList>
          <DialogComboboxOptionListSearch onSearch={(value: string) => setSearchValue(value)}>
            {displayOptions.map((option) => (
              <DialogComboboxOptionListSelectItem
                key={option.value}
                value={option.value}
                onChange={onChange}
                checked={option.value === item?.value}
                css={{
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {option.renderValue()}
              </DialogComboboxOptionListSelectItem>
            ))}
            {/* Currently there's no use case for searching custom values. Eventually we might need it
                  for tags. */}
            {canSearchCustomValue && searchValue && !optionValues.includes(searchValue) ? (
              <DialogComboboxOptionListSelectItem
                key={searchValue}
                value={searchValue}
                onChange={onChange}
                checked={searchValue === item?.value}
                css={{
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                Use "{searchValue}"
              </DialogComboboxOptionListSelectItem>
            ) : null}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

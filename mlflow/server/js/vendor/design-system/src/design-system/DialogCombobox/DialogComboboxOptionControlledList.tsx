import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

import type { DialogComboboxOptionListProps } from './DialogComboboxOptionList';
import { DialogComboboxOptionList } from './DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from './DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSearch } from './DialogComboboxOptionListSearch';
import { DialogComboboxOptionListSelectItem } from './DialogComboboxOptionListSelectItem';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { DialogComboboxOptionListContextProvider } from './providers/DialogComboboxOptionListContext';
import { highlightOption } from './shared';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { EmptyResults, LoadingSpinner } from '../_shared_/Combobox';

export interface DialogComboboxOptionControlledListProps
  extends Omit<DialogComboboxOptionListProps, 'children'>,
    WithLoadingState {
  withSearch?: boolean;
  showAllOption?: boolean;
  allOptionLabel?: string;
  options: string[];
  onChange?: (...args: any[]) => any;
}

export const DialogComboboxOptionControlledList = forwardRef<HTMLDivElement, DialogComboboxOptionControlledListProps>(
  (
    {
      options,
      onChange,
      loading,
      loadingDescription = 'DialogComboboxOptionControlledList',
      withProgressiveLoading,
      withSearch,
      showAllOption,
      allOptionLabel = 'All',
      ...restProps
    },
    forwardedRef,
  ) => {
    const { isInsideDialogCombobox, multiSelect, value, setValue, setIsControlled } = useDialogComboboxContext();
    const [lookAhead, setLookAhead] = useState('');

    if (!isInsideDialogCombobox) {
      throw new Error('`DialogComboboxOptionControlledList` must be used within `DialogCombobox`');
    }

    const lookAheadTimeout = useRef<NodeJS.Timeout | null>(null);
    const ref = useRef<HTMLDivElement>(null);
    useImperativeHandle(forwardedRef, () => ref.current as HTMLDivElement);

    useEffect(() => {
      if (lookAheadTimeout.current) {
        clearTimeout(lookAheadTimeout.current);
      }

      lookAheadTimeout.current = setTimeout(() => {
        setLookAhead('');
      }, 1500);

      return () => {
        if (lookAheadTimeout.current) {
          clearTimeout(lookAheadTimeout.current);
        }
      };
    }, [lookAhead]);

    useEffect(() => {
      if (loading && !withProgressiveLoading) {
        return;
      }

      const optionItems = ref.current?.querySelectorAll('[role="option"]');
      const hasTabIndexedOption = Array.from(optionItems ?? []).some((optionItem) => {
        return optionItem.getAttribute('tabindex') === '0';
      });

      if (!hasTabIndexedOption) {
        const firstOptionItem = optionItems?.[0];
        if (firstOptionItem) {
          highlightOption(firstOptionItem as HTMLElement, undefined, false);
        }
      }
    }, [loading, withProgressiveLoading]);

    const isOptionChecked = options.reduce((acc, option) => {
      acc[option] = value?.includes(option);
      return acc;
    }, {} as Record<string, boolean>);

    const handleUpdate = (updatedValue: string) => {
      setIsControlled(true);

      let newValue = [];
      if (multiSelect) {
        if (value.find((item) => item === updatedValue)) {
          newValue = value.filter((item) => item !== updatedValue);
        } else {
          newValue = [...value, updatedValue];
        }
      } else {
        newValue = [updatedValue];
      }

      setValue(newValue);
      isOptionChecked[updatedValue] = !isOptionChecked[updatedValue];
      if (onChange) {
        onChange(newValue);
      }
    };

    const handleSelectAll = () => {
      setIsControlled(true);

      if (value.length === options.length) {
        setValue([]);
        options.forEach((option) => {
          isOptionChecked[option] = false;
        });
        if (onChange) {
          onChange([]);
        }
      } else {
        setValue(options);
        options.forEach((option) => {
          isOptionChecked[option] = true;
        });
        if (onChange) {
          onChange(options);
        }
      }
    };

    const renderedOptions = (
      <>
        {showAllOption && multiSelect && (
          <DialogComboboxOptionListCheckboxItem
            value="all"
            onChange={handleSelectAll}
            checked={value.length === options.length}
            indeterminate={Boolean(value.length) && value.length !== options.length}
          >
            {allOptionLabel}
          </DialogComboboxOptionListCheckboxItem>
        )}
        {options && options.length > 0 ? (
          options.map((option, key) =>
            multiSelect ? (
              <DialogComboboxOptionListCheckboxItem
                key={key}
                value={option}
                checked={isOptionChecked[option]}
                onChange={handleUpdate}
              >
                {option}
              </DialogComboboxOptionListCheckboxItem>
            ) : (
              <DialogComboboxOptionListSelectItem
                key={key}
                value={option}
                checked={isOptionChecked[option]}
                onChange={handleUpdate}
              >
                {option}
              </DialogComboboxOptionListSelectItem>
            ),
          )
        ) : (
          <EmptyResults />
        )}
      </>
    );

    const optionList = (
      <DialogComboboxOptionList>
        {withSearch ? (
          <DialogComboboxOptionListSearch hasWrapper={true}>{renderedOptions}</DialogComboboxOptionListSearch>
        ) : (
          renderedOptions
        )}
      </DialogComboboxOptionList>
    );

    return (
      <div
        ref={ref}
        aria-busy={loading}
        css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', width: '100%' }}
        {...restProps}
      >
        <DialogComboboxOptionListContextProvider
          value={{ isInsideDialogComboboxOptionList: true, lookAhead, setLookAhead }}
        >
          <>
            {loading ? (
              withProgressiveLoading ? (
                <>
                  {optionList}
                  <LoadingSpinner aria-label="Loading" alt="Loading spinner" loadingDescription={loadingDescription} />
                </>
              ) : (
                <LoadingSpinner aria-label="Loading" alt="Loading spinner" loadingDescription={loadingDescription} />
              )
            ) : (
              optionList
            )}
          </>
        </DialogComboboxOptionListContextProvider>
      </div>
    );
  },
);

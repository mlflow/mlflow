import type { MouseEventHandler } from 'react';
import React, { Children, forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { DialogComboboxOptionListContextProvider } from './providers/DialogComboboxOptionListContext';
import { highlightFirstNonDisabledOption } from './shared';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { EmptyResults, LoadingSpinner } from '../_shared_/Combobox';
import type { HTMLDataAttributes } from '../types';

export interface DialogComboboxOptionListProps
  extends HTMLDataAttributes,
    React.HTMLAttributes<HTMLDivElement>,
    WithLoadingState {
  children: any;
  loading?: boolean;
  withProgressiveLoading?: boolean;
}

export const DialogComboboxOptionList = forwardRef<HTMLDivElement, DialogComboboxOptionListProps>(
  (
    { children, loading, loadingDescription = 'DialogComboboxOptionList', withProgressiveLoading, ...restProps },
    forwardedRef,
  ) => {
    const { isInsideDialogCombobox } = useDialogComboboxContext();

    const ref = useRef<HTMLDivElement>(null);
    useImperativeHandle(forwardedRef, () => ref.current as HTMLDivElement);

    const [lookAhead, setLookAhead] = useState('');

    if (!isInsideDialogCombobox) {
      throw new Error('`DialogComboboxOptionList` must be used within `DialogCombobox`');
    }

    const lookAheadTimeout = useRef<NodeJS.Timeout | null>(null);

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
          highlightFirstNonDisabledOption(firstOptionItem, 'start');
        }
      }
    }, [loading, withProgressiveLoading]);

    const handleOnMouseEnter: MouseEventHandler<HTMLDivElement> = (event) => {
      const target = event.target as HTMLElement;
      if (target) {
        const options = target.hasAttribute('data-combobox-option-list')
          ? target.querySelectorAll('[role="option"]')
          : target?.closest('[data-combobox-option-list="true"]')?.querySelectorAll('[role="option"]');

        if (options) {
          options.forEach((option) => option.removeAttribute('data-highlighted'));
        }
      }
    };

    return (
      <div
        ref={ref}
        aria-busy={loading}
        data-combobox-option-list="true"
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-start',
          width: '100%',
        }}
        onMouseEnter={handleOnMouseEnter}
        {...restProps}
      >
        <DialogComboboxOptionListContextProvider
          value={{ isInsideDialogComboboxOptionList: true, lookAhead, setLookAhead }}
        >
          {loading ? (
            withProgressiveLoading ? (
              <>
                {children}
                <LoadingSpinner aria-label="Loading" alt="Loading spinner" loadingDescription={loadingDescription} />
              </>
            ) : (
              <LoadingSpinner aria-label="Loading" alt="Loading spinner" loadingDescription={loadingDescription} />
            )
          ) : children && Children.toArray(children).some((child) => React.isValidElement(child)) ? (
            children
          ) : (
            <EmptyResults />
          )}
        </DialogComboboxOptionListContextProvider>
      </div>
    );
  },
);

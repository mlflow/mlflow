import { useFloating, flip, shift, autoUpdate, offset, type ExtendedRefs } from '@floating-ui/react';
import React, { forwardRef } from 'react';

import type { ComboboxStateAnalyticsReturnValue } from './hooks';
import { TypeaheadComboboxContextProvider } from './providers/TypeaheadComboboxContext';
import { DesignSystemEventProviderComponentTypes } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';

export interface TypeaheadComboboxRootProps<T> extends React.HTMLAttributes<HTMLDivElement> {
  comboboxState: ComboboxStateAnalyticsReturnValue<T>;
  multiSelect?: boolean;
  floatingUiRefs?: ExtendedRefs<Element>;
  floatingStyles?: React.CSSProperties;
  children: React.ReactNode;
}

export const TypeaheadComboboxRoot: React.FC<TypeaheadComboboxRootProps<any>> = forwardRef<
  HTMLDivElement,
  TypeaheadComboboxRootProps<any>
>(({ comboboxState, multiSelect = false, children, ...props }, ref) => {
  const { classNamePrefix } = useDesignSystemTheme();
  const { refs, floatingStyles } = useFloating({
    whileElementsMounted: autoUpdate,
    middleware: [offset(4), flip(), shift()],
    placement: 'bottom-start',
  });

  return (
    <TypeaheadComboboxContextProvider
      value={{
        componentId: comboboxState.componentId,
        multiSelect,
        isInsideTypeaheadCombobox: true,
        floatingUiRefs: refs,
        floatingStyles: floatingStyles,
      }}
    >
      <div
        {...comboboxState.getComboboxProps({}, { suppressRefError: true })}
        className={`${classNamePrefix}-typeahead-combobox`}
        css={{ display: 'inline-block', width: '100%' }}
        {...props}
        ref={ref}
        data-component-type={DesignSystemEventProviderComponentTypes.TypeaheadCombobox}
        data-component-id={comboboxState.componentId}
      >
        {children}
      </div>
    </TypeaheadComboboxContextProvider>
  );
});

export default TypeaheadComboboxRoot;

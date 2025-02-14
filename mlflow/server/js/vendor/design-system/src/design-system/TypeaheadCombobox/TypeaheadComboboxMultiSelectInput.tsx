import { css, type SerializedStyles } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import type { UseMultipleSelectionReturnValue } from 'downshift';
import { forwardRef, useState, useRef, useLayoutEffect, useEffect } from 'react';

import { CountBadge } from './CountBadge';
import { TypeaheadComboboxControls } from './TypeaheadComboboxControls';
import type { TypeaheadComboboxInputProps } from './TypeaheadComboboxInput';
import { TypeaheadComboboxSelectedItem } from './TypeaheadComboboxSelectedItem';
import { useTypeaheadComboboxContext } from './hooks';
import type { ComboboxStateAnalyticsReturnValue } from './hooks';
import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { LegacyTooltip } from '../LegacyTooltip';
import type { ValidationState } from '../types';
import { getValidationStateColor } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface TypeaheadComboboxMultiSelectInputProps<T> extends TypeaheadComboboxInputProps<T> {
  comboboxState: ComboboxStateAnalyticsReturnValue<T>;
  multipleSelectionState: UseMultipleSelectionReturnValue<T>;
  selectedItems: T[];
  setSelectedItems: React.Dispatch<React.SetStateAction<T[]>>;
  getSelectedItemLabel: (item: T) => React.ReactNode;
  allowClear?: boolean;
  showTagAfterValueCount?: number;
  width?: string | number;
  maxHeight?: string | number;
  disableTooltip?: boolean;
}

const getContainerStyles = (
  theme: Theme,
  validationState?: ValidationState,
  width?: string | number,
  maxHeight?: string | number,
  disabled?: boolean,
): SerializedStyles => {
  const validationColor = getValidationStateColor(theme, validationState);

  return css({
    cursor: 'text',
    display: 'inline-block',
    verticalAlign: 'top',
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.general.borderRadiusBase,
    minHeight: 32,
    height: 'auto',
    minWidth: 0,
    ...(width ? { width } : {}),
    ...(maxHeight ? { maxHeight } : {}),
    padding: '5px 52px 5px 12px',
    position: 'relative',
    overflow: 'auto',
    textOverflow: 'ellipsis',

    '&:hover': {
      border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
    },

    '&:focus-within': {
      outlineColor: theme.colors.actionDefaultBorderFocus,
      outlineWidth: 2,
      outlineOffset: -2,
      outlineStyle: 'solid',
      boxShadow: 'none',
      borderColor: 'transparent',
    },

    '&&': {
      ...(validationState && { borderColor: validationColor }),

      '&:hover': {
        borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover,
      },

      '&:focus': {
        outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        boxShadow: 'none',
        borderColor: 'transparent',
      },

      ...(disabled && {
        borderColor: theme.colors.actionDisabledBorder,
        backgroundColor: theme.colors.actionDisabledBackground,
        cursor: 'not-allowed',
        outline: 'none',

        '&:hover': {
          border: `1px solid ${theme.colors.actionDisabledBorder}`,
        },

        '&:focus-within': {
          outline: 'none',
          borderColor: theme.colors.actionDisabledBorder,
        },
      }),
    },
  });
};

const getContentWrapperStyles = (): SerializedStyles => {
  return css({
    display: 'flex',
    flex: 'auto',
    flexWrap: 'wrap',
    maxWidth: '100%',
    position: 'relative',
  });
};

const getInputWrapperStyles = (): SerializedStyles => {
  return css({
    display: 'inline-flex',
    position: 'relative',
    maxWidth: '100%',
    alignSelf: 'auto',
    flex: 'none',
  });
};

const getInputStyles = (theme: Theme): SerializedStyles => {
  return css({
    lineHeight: 20,
    height: 24,
    margin: 0,
    padding: 0,
    appearance: 'none',
    cursor: 'auto',
    width: '100%',
    backgroundColor: 'transparent',
    color: theme.colors.textPrimary,

    '&, &:hover, &:focus-visible': {
      border: 'none',
      outline: 'none',
    },

    '&::placeholder': {
      color: theme.colors.textPlaceholder,
    },
  });
};

export const TypeaheadComboboxMultiSelectInput = forwardRef<
  HTMLInputElement,
  TypeaheadComboboxMultiSelectInputProps<any>
>(
  (
    {
      comboboxState,
      multipleSelectionState,
      selectedItems,
      setSelectedItems,
      getSelectedItemLabel,
      allowClear = true,
      showTagAfterValueCount = 20,
      width,
      maxHeight,
      placeholder,
      validationState,
      showComboboxToggleButton,
      disableTooltip = false,
      ...restProps
    },
    ref,
  ) => {
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
      throw new Error('`TypeaheadComboboxMultiSelectInput` must be used within `TypeaheadCombobox`');
    }

    const { getInputProps, getToggleButtonProps, toggleMenu, inputValue, setInputValue } = comboboxState;
    const { getSelectedItemProps, getDropdownProps, reset, removeSelectedItem } = multipleSelectionState;
    const { ref: downshiftRef, ...downshiftProps } = getInputProps(getDropdownProps({}, { suppressRefError: true }));
    const {
      floatingUiRefs,
      setInputWidth: setContextInputWidth,
      inputWidth: contextInputWidth,
    } = useTypeaheadComboboxContext();
    const containerRef = useRef<HTMLDivElement>(null);
    const mergedContainerRef = useMergeRefs([containerRef, floatingUiRefs?.setReference]);

    const itemsRef = useRef<HTMLDivElement>(null);
    const measureRef = useRef<HTMLSpanElement>(null);
    const innerRef = useRef<HTMLInputElement>(null);
    const mergedInputRef = useMergeRefs([ref, innerRef, downshiftRef]);

    const { theme } = useDesignSystemTheme();
    const [inputWidth, setInputWidth] = useState(0);

    const shouldShowCountBadge = selectedItems.length > showTagAfterValueCount;
    const [showTooltip, setShowTooltip] = useState<boolean>(shouldShowCountBadge);
    const selectedItemsToRender = selectedItems.slice(0, showTagAfterValueCount);

    const handleClick = () => {
      if (!restProps.disabled) {
        innerRef.current?.focus();
        toggleMenu();
      }
    };

    const handleClear = () => {
      setInputValue('');
      reset();
      setSelectedItems([]);
    };

    // We measure width and set to the input immediately
    useLayoutEffect(() => {
      if (measureRef?.current) {
        const measuredWidth = measureRef.current.scrollWidth;
        setInputWidth(measuredWidth);
      }
    }, [measureRef?.current?.scrollWidth, selectedItems?.length]);

    // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
    useEffect(() => {
      // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
      if (floatingUiRefs?.domReference) {
        const width = floatingUiRefs.domReference.current?.getBoundingClientRect().width ?? 0;
        // Only update context width when the input width updated
        if (width !== contextInputWidth) {
          setContextInputWidth?.(width);
        }
      }
    }, [floatingUiRefs?.domReference, setContextInputWidth, contextInputWidth]);

    // Determine whether to show tooltip
    useEffect(() => {
      let isPartiallyHidden = false;
      if (itemsRef.current && containerRef.current) {
        const { clientHeight: innerHeight } = itemsRef.current;
        const { clientHeight: outerHeight } = containerRef.current;
        isPartiallyHidden = innerHeight > outerHeight;
      }
      setShowTooltip(!disableTooltip && (shouldShowCountBadge || isPartiallyHidden));
    }, [shouldShowCountBadge, itemsRef.current?.clientHeight, containerRef.current?.clientHeight, disableTooltip]);

    const content = (
      <div
        {...addDebugOutlineIfEnabled()}
        onClick={handleClick}
        ref={mergedContainerRef}
        css={getContainerStyles(theme, validationState, width, maxHeight, restProps.disabled)}
        tabIndex={restProps.disabled ? -1 : 0}
      >
        <div ref={itemsRef} css={getContentWrapperStyles()}>
          {selectedItemsToRender?.map((selectedItemForRender, index) => (
            <TypeaheadComboboxSelectedItem
              key={`selected-item-${index}`}
              label={getSelectedItemLabel(selectedItemForRender)}
              item={selectedItemForRender}
              getSelectedItemProps={getSelectedItemProps}
              removeSelectedItem={removeSelectedItem}
              disabled={restProps.disabled}
            />
          ))}
          {shouldShowCountBadge && (
            <CountBadge
              countStartAt={showTagAfterValueCount}
              totalCount={selectedItems.length}
              role="status"
              aria-label="Selected options count"
              disabled={restProps.disabled}
            />
          )}
          <div css={getInputWrapperStyles()}>
            <input
              {...downshiftProps}
              ref={mergedInputRef}
              css={[getInputStyles(theme), { width: inputWidth }]}
              placeholder={selectedItems?.length ? undefined : placeholder}
              aria-controls={comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined}
              {...restProps}
            />
            {/* Since the input element's width is set to be as small as possible in order for the container component to also fit the selected item tags,
             * the span below is used as a "measure node", which contains the same implementation and logic as rc-select.
             * The measure node is visually hidden, and its purpose is to measure the width of the input's current contents
             * (value or placeholder text) and sets the input's width accordingly. This is done in the `useLayoutEffect` hook above.
             */}
            <span ref={measureRef} aria-hidden css={{ visibility: 'hidden', whiteSpace: 'pre', position: 'absolute' }}>
              {innerRef.current?.value ? innerRef.current.value : placeholder}&nbsp;
            </span>
          </div>
        </div>
        <TypeaheadComboboxControls
          getDownshiftToggleButtonProps={getToggleButtonProps}
          showComboboxToggleButton={showComboboxToggleButton}
          showClearSelectionButton={
            allowClear && (Boolean(inputValue) || (selectedItems && selectedItems.length > 0)) && !restProps.disabled
          }
          handleClear={handleClear}
          disabled={restProps.disabled}
        />
      </div>
    );

    if (showTooltip && selectedItems.length > 0) {
      return (
        <LegacyTooltip title={selectedItems.map((item) => getSelectedItemLabel(item)).join(', ')}>
          {content}
        </LegacyTooltip>
      );
    }
    return content;
  },
);

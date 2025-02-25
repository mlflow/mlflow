import React, { Children, forwardRef, useEffect, useRef } from 'react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDialogComboboxOptionListContext } from './hooks/useDialogComboboxOptionListContext';
import { findClosestOptionSibling, findHighlightedOption, getContentOptions, highlightOption } from './shared';
import { useDesignSystemTheme } from '../Hooks';
import { SearchIcon } from '../Icon';
import type { InputProps, InputRef } from '../Input';
import { Input } from '../Input';
import { EmptyResults } from '../_shared_/Combobox';

const extractTextContent = (node: string | number | React.ReactNode): string => {
  if (typeof node === 'string' || typeof node === 'number') {
    return node.toString();
  }

  if (React.isValidElement(node) && node.props.children) {
    return React.Children.toArray(node.props.children).map(extractTextContent).join(' ');
  }

  return '';
};

const filterChildren = (children: React.ReactNode, searchValue: string) => {
  const lowerCaseSearchValue = searchValue.toLowerCase();

  return React.Children.map(children, (child) => {
    if (React.isValidElement(child)) {
      const childType = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']?.defaultProps._TYPE ?? child.props._TYPE;

      if (childType === 'DialogComboboxOptionListSelectItem' || childType === 'DialogComboboxOptionListCheckboxItem') {
        const childTextContent = extractTextContent(child).toLowerCase();
        const childValue = child.props.value?.toLowerCase() ?? '';
        return childTextContent.includes(lowerCaseSearchValue) || childValue.includes(lowerCaseSearchValue)
          ? child
          : null;
      }
    }
    return child;
  })?.filter((child: any) => child);
};

export interface DialogComboboxOptionListSearchProps extends Omit<InputProps, 'componentId' | 'analyticsEvents'> {
  children: any;
  hasWrapper?: boolean;
  virtualized?: boolean;
  onSearch?: (value: string) => void;
  /** Set controlledValue and setControlledValue if search input is controlled, e.g. for custom filtering logic */
  controlledValue?: string;
  setControlledValue?: (value: string) => void;
}

export const DialogComboboxOptionListSearch = forwardRef<InputRef, DialogComboboxOptionListSearchProps>(
  (
    { onChange, onSearch, virtualized, children, hasWrapper, controlledValue, setControlledValue, ...restProps },
    forwardedRef,
  ) => {
    const { theme } = useDesignSystemTheme();
    const { componentId } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList } = useDialogComboboxOptionListContext();

    const [searchValue, setSearchValue] = React.useState<string>();

    if (!isInsideDialogComboboxOptionList) {
      throw new Error('`DialogComboboxOptionListSearch` must be used within `DialogComboboxOptionList`');
    }

    const handleOnChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      if (!virtualized) {
        setSearchValue(event.target.value.toLowerCase());
      }
      setControlledValue?.(event.target.value);
      onSearch?.(event.target.value);
    };

    let filteredChildren = children;

    if (searchValue && !virtualized && controlledValue === undefined) {
      filteredChildren = filterChildren(hasWrapper ? children.props.children : children, searchValue);

      if (hasWrapper) {
        filteredChildren = React.cloneElement(children, {}, filteredChildren);
      }
    }

    const inputWrapperRef = useRef<HTMLDivElement>(null);

    // When the search value changes, highlight the first option
    useEffect(() => {
      if (!inputWrapperRef.current) {
        return;
      }
      const optionItems = getContentOptions(inputWrapperRef.current);

      if (optionItems) {
        // Reset previous highlights
        const highlightedOption = findHighlightedOption(optionItems);

        const firstOptionItem = optionItems?.[0];
        if (firstOptionItem) {
          highlightOption(firstOptionItem, highlightedOption, false);
        }
      }
    }, [searchValue]);

    const handleOnKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Enter') {
        event.preventDefault();
      } else {
        return;
      }

      // Find closest parent of type DialogComboboxOptionList and get all options within it
      const options = getContentOptions(event.target as HTMLElement);

      if (!options) {
        return;
      }

      const highlightedOption = findHighlightedOption(options);

      // If the user is navigating the option is highlighted without focusing in order to avoid losing focus on the input
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        if (highlightedOption) {
          const nextOption = findClosestOptionSibling(
            highlightedOption,
            event.key === 'ArrowDown' ? 'next' : 'previous',
          );

          if (nextOption) {
            highlightOption(nextOption, highlightedOption, false);
          } else if (event.key === 'ArrowDown') {
            // If there is no next option, highlight the first option
            const firstOption = options[0];
            highlightOption(firstOption, highlightedOption, false);
          } else if (event.key === 'ArrowUp') {
            // If there is no previous option, highlight the last option
            const lastOption = options[options.length - 1];
            highlightOption(lastOption, highlightedOption, false);
          }
        } else {
          // In case there is no highlighted option, highlight the first / last option depending on key
          const nextOption = event.key === 'ArrowDown' ? options[0] : options[options.length - 1];
          if (nextOption) {
            highlightOption(nextOption, undefined, false);
          }
        }
        // On Enter trigger a click event on the highlighted option
      } else if (event.key === 'Enter' && highlightedOption) {
        highlightedOption.click();
      }
    };

    const childrenIsNotEmpty = Children.toArray(hasWrapper ? children.props.children : children).some((child) =>
      React.isValidElement(child),
    );

    return (
      <>
        <div
          ref={inputWrapperRef}
          css={{
            padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px`,
            width: '100%',
            boxSizing: 'border-box',
            position: 'sticky',
            top: 0,
            background: theme.colors.backgroundPrimary,
            zIndex: theme.options.zIndexBase + 1,
          }}
        >
          <Input
            componentId={
              componentId
                ? `${componentId}.search`
                : 'codegen_design_system_src_design_system_dialogcombobox_dialogcomboboxoptionlistsearch.tsx_173'
            }
            type="search"
            name="search"
            ref={forwardedRef}
            prefix={<SearchIcon />}
            placeholder="Search"
            onChange={handleOnChange}
            onKeyDown={(event) => {
              handleOnKeyDown(event);
              restProps.onKeyDown?.(event);
            }}
            value={controlledValue ?? searchValue}
            {...restProps}
          />
        </div>
        {virtualized ? (
          children
        ) : ((hasWrapper && filteredChildren?.props.children?.length) || (!hasWrapper && filteredChildren?.length)) &&
          childrenIsNotEmpty ? (
          <div
            aria-live="polite"
            css={{
              width: '100%',
            }}
          >
            {filteredChildren}
          </div>
        ) : (
          <EmptyResults />
        )}
      </>
    );
  },
);

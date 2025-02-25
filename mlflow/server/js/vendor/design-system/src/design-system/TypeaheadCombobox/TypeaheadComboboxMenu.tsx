import { css } from '@emotion/react';
import type { SerializedStyles } from '@emotion/react';
import { computePosition, flip, size } from '@floating-ui/dom';
import { useMergeRefs } from '@floating-ui/react';
import type { UseComboboxReturnValue } from 'downshift';
import React, { Children, Fragment, forwardRef, useCallback, useEffect, useState } from 'react';
import type { HTMLAttributes, ReactNode } from 'react';
import { createPortal } from 'react-dom';

import { useTypeaheadComboboxContext } from './hooks';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { EmptyResults, LoadingSpinner, getComboboxContentWrapperStyles } from '../_shared_/Combobox';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface TypeaheadComboboxMenuProps<T> extends HTMLAttributes<HTMLUListElement> {
  comboboxState: UseComboboxReturnValue<T>;
  loading?: boolean;
  emptyText?: string | React.ReactNode;
  width?: number | string;
  minWidth?: number | string;
  maxWidth?: number | string;
  minHeight?: number | string;
  maxHeight?: number | string;
  /* For use with react-virtual: pass in virtualizer.totalSize */
  listWrapperHeight?: number;
  /* For use with react-virtual: pass in parentRef */
  virtualizerRef?: React.RefObject<T>;
  children?: ReactNode;
  matchTriggerWidth?: boolean;
}

const getTypeaheadComboboxMenuStyles = (): SerializedStyles => {
  return css({
    padding: 0,
    margin: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    position: 'absolute',
  });
};

export const TypeaheadComboboxMenu = forwardRef<HTMLElement | null, TypeaheadComboboxMenuProps<any>>(
  (
    {
      comboboxState,
      loading,
      emptyText,
      width,
      minWidth = 240,
      maxWidth,
      minHeight,
      maxHeight,
      listWrapperHeight,
      virtualizerRef,
      children,
      matchTriggerWidth,
      ...restProps
    },
    ref,
  ) => {
    const { getMenuProps, isOpen } = comboboxState;
    const { ref: downshiftRef, ...downshiftProps } = getMenuProps({}, { suppressRefError: true });
    const { useNewShadows } = useDesignSystemSafexFlags();
    const [viewPortMaxHeight, setViewPortMaxHeight] = useState<number | undefined>(undefined);
    const { floatingUiRefs, floatingStyles, isInsideTypeaheadCombobox, inputWidth } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
      throw new Error('`TypeaheadComboboxMenu` must be used within `TypeaheadCombobox`');
    }

    const mergedRef = useMergeRefs([ref, downshiftRef, floatingUiRefs?.setFloating, virtualizerRef]);
    const { theme } = useDesignSystemTheme();
    const { getPopupContainer } = useDesignSystemContext();

    const recalculateMaxHeight = useCallback(() => {
      if (
        isOpen &&
        floatingUiRefs?.floating &&
        floatingUiRefs.reference.current &&
        floatingUiRefs?.reference &&
        floatingUiRefs.floating.current
      ) {
        computePosition(floatingUiRefs.reference.current, floatingUiRefs.floating.current, {
          middleware: [
            flip(),
            size({
              padding: theme.spacing.sm,
              apply({ availableHeight }) {
                setViewPortMaxHeight(availableHeight);
              },
            }),
          ],
        });
      }
    }, [isOpen, floatingUiRefs, theme.spacing.sm]);

    useEffect(() => {
      if (isOpen && !maxHeight) {
        window.addEventListener('scroll', recalculateMaxHeight);

        return () => {
          window.removeEventListener('scroll', recalculateMaxHeight);
        };
      } else {
        return;
      }
    }, [isOpen, maxHeight, recalculateMaxHeight]);

    if (!isOpen) return null;

    const hasFragmentWrapper = children && !Array.isArray(children) && (children as any).type === Fragment;
    const filterableChildren = hasFragmentWrapper ? (children as any).props.children : children;

    const hasResults =
      filterableChildren &&
      Children.toArray(filterableChildren).some((child) => {
        if (React.isValidElement(child)) {
          const childType = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']?.defaultProps._TYPE ?? child.props._TYPE;
          return ['TypeaheadComboboxMenuItem', 'TypeaheadComboboxCheckboxItem'].includes(childType);
        }
        return false;
      });

    const [menuItemChildren, footer] = Children.toArray(children).reduce<React.ReactNode[][]>(
      (acc, child) => {
        const isFooter = React.isValidElement(child) && child.props._TYPE === 'TypeaheadComboboxFooter';
        if (isFooter) {
          acc[1].push(child);
        } else {
          acc[0].push(child);
        }
        return acc;
      },
      [[], []],
    );

    return createPortal(
      <ul
        {...addDebugOutlineIfEnabled()}
        aria-busy={loading}
        {...downshiftProps}
        ref={mergedRef}
        css={[
          getComboboxContentWrapperStyles(theme, {
            maxHeight: maxHeight ?? viewPortMaxHeight,
            maxWidth,
            minHeight,
            minWidth,
            width,
            useNewShadows,
          }),
          getTypeaheadComboboxMenuStyles(),
          matchTriggerWidth && inputWidth && { width: inputWidth },
        ]}
        style={{ ...floatingStyles }}
        {...restProps}
      >
        {loading ? (
          <LoadingSpinner aria-label="Loading" alt="Loading spinner" />
        ) : hasResults ? (
          <>
            <div
              style={{
                position: 'relative',
                width: '100%',
                ...(listWrapperHeight && { height: listWrapperHeight, flexShrink: 0 }),
              }}
            >
              {menuItemChildren}
            </div>
            {footer}
          </>
        ) : (
          <>
            <EmptyResults emptyText={emptyText} />
            {footer}
          </>
        )}
      </ul>,
      getPopupContainer ? getPopupContainer() : document.body,
    );
  },
);

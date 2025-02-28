import * as Popover from '@radix-ui/react-popover';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import { useModalContext } from '../Modal';
import { EmptyResults, LoadingSpinner, getComboboxContentWrapperStyles } from '../_shared_/Combobox';
import type { HTMLDataAttributes } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface DialogComboboxContentProps extends Popover.PopoverContentProps, HTMLDataAttributes, WithLoadingState {
  width?: number | string;
  loading?: boolean;
  maxHeight?: number;
  maxWidth?: number;
  minHeight?: number;
  minWidth?: number;
  side?: 'top' | 'bottom';
  matchTriggerWidth?: boolean;
  textOverflowMode?: 'ellipsis' | 'multiline';
  forceCloseOnEscape?: boolean;
}

const defaultMaxHeight = 'var(--radix-popover-content-available-height)';

export const DialogComboboxContent = forwardRef<HTMLDivElement, DialogComboboxContentProps>(
  (
    {
      children,
      loading,
      loadingDescription = 'DialogComboboxContent',
      matchTriggerWidth,
      textOverflowMode,
      maxHeight,
      maxWidth,
      minHeight,
      minWidth = 240,
      width,
      align = 'start',
      side = 'bottom',
      sideOffset = 4,
      onEscapeKeyDown,
      onKeyDown,
      forceCloseOnEscape,
      ...restProps
    },
    forwardedRef,
  ) => {
    const { theme } = useDesignSystemTheme();
    const {
      label,
      isInsideDialogCombobox,
      contentWidth,
      setContentWidth,
      textOverflowMode: contextTextOverflowMode,
      setTextOverflowMode,
      multiSelect,
      isOpen,
      rememberLastScrollPosition,
      setIsOpen,
    } = useDialogComboboxContext();
    const { isInsideModal } = useModalContext();
    const { getPopupContainer } = useDesignSystemContext();
    const { useNewShadows } = useDesignSystemSafexFlags();
    const [lastScrollPosition, setLastScrollPosition] = useState<number>(0);
    if (!isInsideDialogCombobox) {
      throw new Error('`DialogComboboxContent` must be used within `DialogCombobox`');
    }

    const contentRef = useRef<HTMLDivElement>(null);
    useImperativeHandle(forwardedRef, () => contentRef.current as HTMLDivElement);
    const realContentWidth = matchTriggerWidth ? 'var(--radix-popover-trigger-width)' : width;

    useEffect(() => {
      if (rememberLastScrollPosition) {
        if (!isOpen && contentRef.current) {
          setLastScrollPosition(contentRef.current.scrollTop);
        } else {
          // Wait for the popover to render and scroll to the last scrolled position
          const interval = setInterval(() => {
            if (contentRef.current) {
              // Verify if the popover's content can be scrolled to the last scrolled position
              if (lastScrollPosition && contentRef.current.scrollHeight >= lastScrollPosition) {
                contentRef.current.scrollTo({ top: lastScrollPosition, behavior: 'smooth' });
              }
              clearInterval(interval);
            }
          }, 50);

          return () => clearInterval(interval);
        }
      }

      return;
    }, [isOpen, rememberLastScrollPosition, lastScrollPosition]);

    useEffect(() => {
      if (contentWidth !== realContentWidth) {
        setContentWidth(realContentWidth);
      }
    }, [realContentWidth, contentWidth, setContentWidth]);

    useEffect(() => {
      if (textOverflowMode !== contextTextOverflowMode) {
        setTextOverflowMode(textOverflowMode ? textOverflowMode : 'multiline');
      }
    }, [textOverflowMode, contextTextOverflowMode, setTextOverflowMode]);

    return (
      <Popover.Portal container={getPopupContainer && getPopupContainer()}>
        <Popover.Content
          {...addDebugOutlineIfEnabled()}
          aria-label={`${label} options`}
          aria-busy={loading}
          role="listbox"
          aria-multiselectable={multiSelect}
          css={getComboboxContentWrapperStyles(theme, {
            maxHeight: maxHeight ? `min(${maxHeight}px, ${defaultMaxHeight})` : defaultMaxHeight,
            maxWidth,
            minHeight,
            minWidth,
            width: realContentWidth,
            useNewShadows,
          })}
          align={align}
          side={side}
          sideOffset={sideOffset}
          onKeyDown={(e) => {
            // This is a workaround for Radix's DialogCombobox.Content not receiving Escape key events
            // when nested inside a modal. We need to stop propagation of the event so that the modal
            // doesn't close when the DropdownMenu should.
            if (e.key === 'Escape') {
              if (isInsideModal || forceCloseOnEscape) {
                e.stopPropagation();
                setIsOpen(false);
              }
              onEscapeKeyDown?.(e.nativeEvent);
            }
            onKeyDown?.(e);
          }}
          {...restProps}
          ref={contentRef}
        >
          <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' }}>
            {loading ? (
              <LoadingSpinner label="Loading" alt="Loading spinner" loadingDescription={loadingDescription} />
            ) : children ? (
              children
            ) : (
              <EmptyResults />
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    );
  },
);

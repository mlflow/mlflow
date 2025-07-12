import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import * as Popover from '@radix-ui/react-popover';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { useModalContext } from '../Modal';
import { EmptyResults, LoadingSpinner, getComboboxContentWrapperStyles } from '../_shared_/Combobox';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const defaultMaxHeight = 'var(--radix-popover-content-available-height)';
export const DialogComboboxContent = forwardRef(({ children, loading, loadingDescription = 'DialogComboboxContent', matchTriggerWidth, textOverflowMode, maxHeight, maxWidth, minHeight, minWidth = 240, width, align = 'start', side = 'bottom', sideOffset = 4, onEscapeKeyDown, onKeyDown, forceCloseOnEscape, ...restProps }, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const { label, isInsideDialogCombobox, contentWidth, setContentWidth, textOverflowMode: contextTextOverflowMode, setTextOverflowMode, multiSelect, isOpen, rememberLastScrollPosition, setIsOpen, } = useDialogComboboxContext();
    const { isInsideModal } = useModalContext();
    const { getPopupContainer } = useDesignSystemContext();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const [lastScrollPosition, setLastScrollPosition] = useState(0);
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxContent` must be used within `DialogCombobox`');
    }
    const contentRef = useRef(null);
    useImperativeHandle(forwardedRef, () => contentRef.current);
    const realContentWidth = matchTriggerWidth ? 'var(--radix-popover-trigger-width)' : width;
    useEffect(() => {
        if (rememberLastScrollPosition) {
            if (!isOpen && contentRef.current) {
                setLastScrollPosition(contentRef.current.scrollTop);
            }
            else {
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
    return (_jsx(Popover.Portal, { container: getPopupContainer && getPopupContainer(), children: _jsx(Popover.Content, { ...addDebugOutlineIfEnabled(), "aria-label": `${label} options`, "aria-busy": loading, role: "listbox", "aria-multiselectable": multiSelect, css: getComboboxContentWrapperStyles(theme, {
                maxHeight: maxHeight ? `min(${maxHeight}px, ${defaultMaxHeight})` : defaultMaxHeight,
                maxWidth,
                minHeight,
                minWidth,
                width: realContentWidth,
                useNewShadows,
                useNewBorderColors,
            }), align: align, side: side, sideOffset: sideOffset, onKeyDown: (e) => {
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
            }, ...restProps, ref: contentRef, children: _jsx("div", { css: { display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' }, children: loading ? (_jsx(LoadingSpinner, { label: "Loading", alt: "Loading spinner", loadingDescription: loadingDescription })) : children ? (children) : (_jsx(EmptyResults, {})) }) }) }));
});
//# sourceMappingURL=DialogComboboxContent.js.map
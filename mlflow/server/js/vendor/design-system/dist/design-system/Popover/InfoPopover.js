import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { useState } from 'react';
import * as Popover from './Popover';
import { useDesignSystemTheme } from '../Hooks';
import { InfoIcon } from '../Icon';
import { useModalContext } from '../Modal';
export const InfoPopover = ({ children, popoverProps, iconTitle, iconProps, isKeyboardFocusable = true, ariaLabel = 'More details', }) => {
    const { theme } = useDesignSystemTheme();
    const { isInsideModal } = useModalContext();
    const [open, setOpen] = useState(false);
    const handleKeyDown = (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            setOpen(!open);
        }
    };
    const { onKeyDown, ...restPopoverProps } = popoverProps || {};
    return (_jsxs(Popover.Root, { componentId: "codegen_design-system_src_design-system_popover_infopopover.tsx_36", open: open, onOpenChange: setOpen, children: [_jsx(Popover.Trigger, { asChild: true, children: _jsx("span", { style: { display: 'inline-flex', cursor: 'pointer' }, 
                    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                    tabIndex: isKeyboardFocusable ? 0 : -1, onKeyDown: handleKeyDown, "aria-label": iconTitle ? undefined : ariaLabel, role: "button", onClick: (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setOpen(!open);
                    }, children: _jsx(InfoIcon, { "aria-hidden": iconTitle ? false : true, title: iconTitle, "aria-label": iconTitle, css: {
                            color: theme.colors.textSecondary,
                        }, ...iconProps }) }) }), _jsxs(Popover.Content, { align: "start", onKeyDown: (e) => {
                    if (e.key === 'Escape') {
                        // If inside an AntD Modal, stop propagation of Escape key so that the modal doesn't close.
                        // This is specifically for that case, so we only do it if inside a modal to limit the blast radius.
                        if (isInsideModal) {
                            e.stopPropagation();
                            // If stopping propagation, we also need to manually close the popover since the radix
                            // library expects the event to bubble up to the parent components.
                            setOpen(false);
                        }
                    }
                    onKeyDown?.(e);
                }, ...restPopoverProps, children: [children, _jsx(Popover.Arrow, {})] })] }));
};
//# sourceMappingURL=InfoPopover.js.map
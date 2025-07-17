import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { isUndefined, noop } from 'lodash';
import React, { useMemo, useState } from 'react';
import { Button, CloseIcon, InfoIcon, Modal, Sidebar, Tooltip, Typography, useDesignSystemTheme, } from '../../../design-system';
export function Root({ children, initialContentId }) {
    const [currentContentId, setCurrentContentId] = useState(initialContentId);
    return (_jsx(DocumentationSideBarContext.Provider, { value: useMemo(() => ({ currentContentId, setCurrentContentId }), [currentContentId, setCurrentContentId]), children: children }));
}
const DocumentationSideBarContext = React.createContext({
    currentContentId: undefined,
    setCurrentContentId: noop,
});
export const useDocumentationSidebarContext = () => {
    const context = React.useContext(DocumentationSideBarContext);
    return context;
};
export function Trigger({ contentId, label, tooltipContent, asChild, children, ...tooltipProps }) {
    const { theme } = useDesignSystemTheme();
    const { setCurrentContentId } = useDocumentationSidebarContext();
    const triggerProps = useMemo(() => ({
        onClick: () => setCurrentContentId(contentId),
        [`aria-label`]: label,
    }), [contentId, label, setCurrentContentId]);
    const renderAsChild = asChild && React.isValidElement(children);
    return (_jsx(Tooltip, { ...tooltipProps, content: tooltipContent, children: renderAsChild ? (React.cloneElement(children, triggerProps)) : (_jsx("button", { css: {
                border: 'none',
                backgroundColor: 'transparent',
                padding: 0,
                display: 'flex',
                height: 'var(--spacing-md)',
                alignItems: 'center',
                cursor: 'pointer',
            }, ...triggerProps, children: _jsx(InfoIcon, { css: { fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary } }) })) }));
}
export function Content({ title, modalTitleWhenCompact, width, children, closeLabel, displayModalWhenCompact, }) {
    const { theme } = useDesignSystemTheme();
    const { currentContentId, setCurrentContentId } = useDocumentationSidebarContext();
    if (isUndefined(currentContentId)) {
        return null;
    }
    const content = React.isValidElement(children)
        ? React.cloneElement(children, { contentId: currentContentId })
        : children;
    if (displayModalWhenCompact) {
        return (_jsx(Modal, { componentId: `documentation-side-bar-compact-modal-${currentContentId}`, visible: true, size: "wide", onOk: () => setCurrentContentId(undefined), okText: closeLabel, okButtonProps: { type: undefined }, onCancel: () => setCurrentContentId(undefined), title: modalTitleWhenCompact ?? title, children: content }));
    }
    return (_jsx(Sidebar, { position: "right", dangerouslyAppendEmotionCSS: { border: 'none' }, children: _jsx(Sidebar.Content, { componentId: `documentation-side-bar-content-${currentContentId}`, openPanelId: 0, closable: true, disableResize: true, enableCompact: true, width: width, children: _jsx(Sidebar.Panel, { panelId: 0, children: _jsxs("div", { css: {
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        rowGap: theme.spacing.md,
                        borderRadius: theme.legacyBorders.borderRadiusLg,
                        border: `1px solid ${theme.colors.backgroundSecondary}`,
                        padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
                        backgroundColor: theme.colors.backgroundSecondary,
                    }, children: [_jsxs("div", { css: {
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                width: '100%',
                            }, children: [_jsx(Typography.Text, { color: "secondary", children: title }), _jsx(Button, { "aria-label": closeLabel, icon: _jsx(CloseIcon, {}), componentId: `documentation-side-bar-close-${currentContentId}`, onClick: () => setCurrentContentId(undefined) })] }), content] }) }) }) }));
}
//# sourceMappingURL=DocumentationSidebar.js.map
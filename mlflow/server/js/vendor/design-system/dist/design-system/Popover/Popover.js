import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import * as Popover from '@radix-ui/react-popover';
import { forwardRef, useCallback, useEffect, useMemo, useRef } from 'react';
import { ComponentFinderContext, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useComponentFinderContext, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { useDesignSystemSafexFlags } from '../utils';
import { getDarkModePortalStyles, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
// WRAPPED RADIX-UI-COMPONENTS
export const Anchor = Popover.Anchor; // Behavioral component only
export const Root = ({ children, onOpenChange, componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView], ...props }) => {
    const firstView = useRef(true);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Popover,
        componentId: componentId ?? 'design_system.popover',
        analyticsEvents: memoizedAnalyticsEvents,
    });
    useEffect(() => {
        if (props.open && firstView.current) {
            eventContext.onView();
            firstView.current = false;
        }
    }, [eventContext, props.open]);
    const onOpenChangeHandler = useCallback((open) => {
        if (open && firstView.current) {
            eventContext.onView();
            firstView.current = false;
        }
        onOpenChange?.(open);
    }, [eventContext, onOpenChange]);
    return (_jsx(Popover.Root, { ...props, onOpenChange: onOpenChangeHandler, children: _jsx(ComponentFinderContext.Provider, { value: { dataComponentProps: eventContext.dataComponentProps }, children: children }) }));
};
export const Content = forwardRef(function Content({ children, minWidth = 220, maxWidth, ...props }, ref) {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const dataComponentProps = useComponentFinderContext(DesignSystemEventProviderComponentTypes.Popover);
    return (_jsx(Popover.Portal, { container: getPopupContainer && getPopupContainer(), children: _jsx(Popover.Content, { ...addDebugOutlineIfEnabled(), ref: ref, css: [contentStyles(theme, useNewShadows, useNewBorderColors), { minWidth, maxWidth }], sideOffset: 4, "aria-label": "Popover content", ...props, ...dataComponentProps, children: children }) }));
});
export const Trigger = forwardRef(function Trigger({ children, ...props }, ref) {
    return (_jsx(Popover.Trigger, { ...addDebugOutlineIfEnabled(), ref: ref, ...props, children: children }));
});
export const Close = forwardRef(function Close({ children, ...props }, ref) {
    return (_jsx(Popover.Close, { ref: ref, ...props, children: children }));
});
export const Arrow = forwardRef(function Arrow({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return (_jsx(Popover.Arrow, { css: {
            fill: theme.colors.backgroundPrimary,
            stroke: theme.colors.borderDecorative,
            strokeDashoffset: -CONSTANTS.arrowBottomLength(),
            strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
            strokeWidth: CONSTANTS.arrowStrokeWidth(),
            // TODO: This is a temporary fix for the alignment of the Arrow;
            // Radix has changed the implementation for v1.0.0 (uses floating-ui)
            // which has new behaviors for alignment that we don't want. Generally
            // we need to fix the arrow to always be aligned to the left of the menu (with
            // offset equal to border radius)
            position: 'relative',
            top: -1,
        }, ref: ref, width: 12, height: 6, ...props, children: children }));
});
// CONSTANTS
const CONSTANTS = {
    arrowBottomLength() {
        // The built in arrow is a polygon: 0,0 30,0 15,10
        return 30;
    },
    arrowHeight() {
        return 10;
    },
    arrowSide() {
        return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
    },
    arrowStrokeWidth() {
        // This is eyeballed b/c relative to the svg viewbox coordinate system
        return 2;
    },
};
const popoverContentStyles = (theme, useNewShadows, useNewBorderColors) => ({
    backgroundColor: theme.colors.backgroundPrimary,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
    borderRadius: theme.borders.borderRadiusSm,
    padding: `${theme.spacing.sm}px`,
    boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
    zIndex: theme.options.zIndexBase + 30,
    ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
    a: importantify({
        color: theme.colors.actionTertiaryTextDefault,
        '&:hover, &:focus': {
            color: theme.colors.actionTertiaryTextHover,
        },
    }),
    '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '1px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
    },
});
const contentStyles = (theme, useNewShadows, useNewBorderColors) => ({
    ...popoverContentStyles(theme, useNewShadows, useNewBorderColors),
});
//# sourceMappingURL=Popover.js.map
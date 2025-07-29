import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import * as RadixHoverCard from '@radix-ui/react-hover-card';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { useDesignSystemSafexFlags } from '../utils';
import { getDarkModePortalStyles, importantify } from '../utils/css-utils';
/**
 * The HoverCard component combines Radix's HoverCard primitives into a single, easy-to-use component.
 * It handles the setup of the trigger, content, and arrow elements, as well as applying custom styles
 * using Emotion CSS
 */
export const HoverCard = ({ trigger, content, side = 'top', sideOffset = 4, align = 'center', minWidth = 220, maxWidth, ...props }) => {
    const { getPopupContainer } = useDesignSystemContext();
    const { useNewBorderColors } = useDesignSystemSafexFlags();
    const hoverCardStyles = useHoverCardStyles({ minWidth, maxWidth, useNewBorderColors });
    return (_jsxs(RadixHoverCard.Root, { ...props, children: [_jsx(RadixHoverCard.Trigger, { asChild: true, children: trigger }), _jsx(RadixHoverCard.Portal, { container: getPopupContainer && getPopupContainer(), children: _jsxs(RadixHoverCard.Content, { side: side, sideOffset: sideOffset, align: align, css: hoverCardStyles['content'], children: [content, _jsx(RadixHoverCard.Arrow, { css: hoverCardStyles['arrow'] })] }) })] }));
};
// CONSTANTS used for defining the Arrow's appearance and behavior
const CONSTANTS = {
    arrowWidth: 12,
    arrowHeight: 6,
    arrowBottomLength() {
        // The built in arrow is a polygon: 0,0 30,0 15,10
        return 30;
    },
    arrowSide() {
        return 2 * (this.arrowHeight ** 2 * 2) ** 0.5;
    },
    arrowStrokeWidth() {
        // This is eyeballed b/c relative to the svg viewbox coordinate system
        return 2;
    },
};
/**
 * A custom hook to generate CSS styles for the HoverCard's content and arrow.
 * These styles are dynamically generated based on the theme and optional min/max width props.
 * The hook also applies necessary dark mode adjustments
 */
const useHoverCardStyles = ({ minWidth, maxWidth, useNewBorderColors, }) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();
    return {
        content: {
            backgroundColor: theme.colors.backgroundPrimary,
            color: theme.colors.textPrimary,
            lineHeight: theme.typography.lineHeightBase,
            border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
            borderRadius: theme.borders.borderRadiusSm,
            padding: `${theme.spacing.sm}px`,
            boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
            userSelect: 'none',
            zIndex: theme.options.zIndexBase + 30,
            minWidth,
            maxWidth,
            ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
            a: importantify({
                color: theme.colors.actionTertiaryTextDefault,
                cursor: 'default',
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
        },
        arrow: {
            fill: theme.colors.backgroundPrimary,
            height: CONSTANTS.arrowHeight,
            stroke: theme.colors.borderDecorative,
            strokeDashoffset: -CONSTANTS.arrowBottomLength(),
            strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
            strokeWidth: CONSTANTS.arrowStrokeWidth(),
            width: CONSTANTS.arrowWidth,
            position: 'relative',
            top: -1,
            zIndex: theme.options.zIndexBase + 30,
        },
    };
};
//# sourceMappingURL=HoverCard.js.map
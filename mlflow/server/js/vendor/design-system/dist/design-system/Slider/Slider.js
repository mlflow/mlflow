import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import * as RadixSlider from '@radix-ui/react-slider';
import { forwardRef } from 'react';
import { useDesignSystemSafexFlags } from '..';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getRootStyles = () => {
    return css({
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        '&[data-orientation="vertical"]': {
            flexDirection: 'column',
            width: 20,
            height: 100,
        },
        '&[data-orientation="horizontal"]': {
            height: 20,
            width: 200,
        },
    });
};
export const Root = forwardRef((props, ref) => {
    return _jsx(RadixSlider.Root, { ...addDebugOutlineIfEnabled(), css: getRootStyles(), ...props, ref: ref });
});
const getTrackStyles = (theme) => {
    return css({
        backgroundColor: theme.colors.grey100,
        position: 'relative',
        flexGrow: 1,
        borderRadius: theme.borders.borderRadiusFull,
        '&[data-orientation="vertical"]': {
            width: 3,
        },
        '&[data-orientation="horizontal"]': {
            height: 3,
        },
    });
};
export const Track = forwardRef((props, ref) => {
    const { theme } = useDesignSystemTheme();
    return _jsx(RadixSlider.Track, { css: getTrackStyles(theme), ...props, ref: ref });
});
const getRangeStyles = (theme) => {
    return css({
        backgroundColor: theme.colors.primary,
        position: 'absolute',
        borderRadius: theme.borders.borderRadiusFull,
        height: '100%',
        '&[data-disabled]': {
            backgroundColor: theme.colors.grey100,
        },
    });
};
export const Range = forwardRef((props, ref) => {
    const { theme } = useDesignSystemTheme();
    return _jsx(RadixSlider.Range, { css: getRangeStyles(theme), ...props, ref: ref });
});
const getThumbStyles = (theme, useNewShadows) => {
    return css({
        display: 'block',
        width: 20,
        height: 20,
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        boxShadow: useNewShadows ? theme.shadows.xs : `0 2px 4px 0 ${theme.colors.grey400}`,
        borderRadius: theme.borders.borderRadiusFull,
        outline: 'none',
        '&:hover': {
            backgroundColor: theme.colors.actionPrimaryBackgroundHover,
        },
        '&:focus': {
            backgroundColor: theme.colors.actionPrimaryBackgroundPress,
        },
        '&[data-disabled]': {
            backgroundColor: theme.colors.grey200,
            boxShadow: 'none',
        },
    });
};
export const Thumb = forwardRef((props, ref) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows } = useDesignSystemSafexFlags();
    return (_jsx(RadixSlider.Thumb, { css: getThumbStyles(theme, useNewShadows), "aria-label": "Slider thumb", ...props, ref: ref }));
});
//# sourceMappingURL=Slider.js.map
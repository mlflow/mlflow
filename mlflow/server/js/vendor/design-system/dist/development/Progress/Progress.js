import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import * as Progress from '@radix-ui/react-progress';
import React from 'react';
import { ProgressContext, ProgressContextProvider } from './providers/ProgressContex';
import { importantify, useDesignSystemTheme } from '../../design-system';
const getProgressRootStyles = (theme, { minWidth, maxWidth }) => {
    const styles = {
        position: 'relative',
        overflow: 'hidden',
        backgroundColor: theme.colors.progressTrack,
        height: theme.spacing.sm,
        width: '100%',
        borderRadius: theme.borders.borderRadiusFull,
        ...(minWidth && { minWidth }),
        ...(maxWidth && { maxWidth }),
        /* Fix overflow clipping in Safari */
        /* https://gist.github.com/domske/b66047671c780a238b51c51ffde8d3a0 */
        transform: 'translateZ(0)',
    };
    return css(importantify(styles));
};
export const Root = (props) => {
    const { children, value, minWidth, maxWidth, ...restProps } = props;
    const { theme } = useDesignSystemTheme();
    return (_jsx(ProgressContextProvider, { value: { progress: value }, children: _jsx(Progress.Root, { value: value, ...restProps, css: getProgressRootStyles(theme, { minWidth, maxWidth }), children: children }) }));
};
const getProgressIndicatorStyles = (theme) => {
    const styles = {
        backgroundColor: theme.colors.progressFill,
        height: '100%',
        width: '100%',
        transition: 'transform 300ms linear',
        borderRadius: theme.borders.borderRadiusFull,
    };
    return css(importantify(styles));
};
export const Indicator = (props) => {
    const { progress } = React.useContext(ProgressContext);
    const { theme } = useDesignSystemTheme();
    return (_jsx(Progress.Indicator, { css: getProgressIndicatorStyles(theme), style: { transform: `translateX(-${100 - (progress ?? 100)}%)` }, ...props }));
};
//# sourceMappingURL=Progress.js.map
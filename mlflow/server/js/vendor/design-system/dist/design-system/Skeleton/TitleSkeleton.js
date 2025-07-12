import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingState } from '../LoadingState/LoadingState';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const titleContainerStyles = css({
    cursor: 'progress',
    width: '100%',
    height: 28,
    display: 'flex',
    justifyContent: 'flex-start',
    alignItems: 'center',
});
const titleFillStyles = css({
    borderRadius: 'var(--border-radius)',
    height: 12,
    width: '100%',
});
export const TitleSkeleton = ({ label, frameRate = 60, style, loading = true, loadingDescription = 'TitleSkeleton', ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsxs("div", { ...addDebugOutlineIfEnabled(), css: titleContainerStyles, style: {
            ...style,
            ['--border-radius']: `${theme.general.borderRadiusBase}px`,
        }, ...restProps, children: [loading && _jsx(LoadingState, { description: loadingDescription }), _jsx("span", { css: visuallyHidden, children: label }), _jsx("div", { "aria-hidden": true, css: [titleFillStyles, genSkeletonAnimatedColor(theme, frameRate)] })] }));
};
//# sourceMappingURL=TitleSkeleton.js.map
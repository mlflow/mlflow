import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingState } from '../LoadingState/LoadingState';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const GenericContainerStyles = css({
    cursor: 'progress',
    borderRadius: 'var(--border-radius)',
});
export const GenericSkeleton = ({ label, frameRate = 60, style, loading = true, loadingDescription = 'GenericSkeleton', ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsxs("div", { ...addDebugOutlineIfEnabled(), css: [GenericContainerStyles, genSkeletonAnimatedColor(theme, frameRate)], style: {
            ...style,
            ['--border-radius']: `${theme.general.borderRadiusBase}px`,
        }, ...restProps, children: [loading && _jsx(LoadingState, { description: loadingDescription }), _jsx("span", { css: visuallyHidden, children: label })] }));
};
//# sourceMappingURL=GenericSkeleton.js.map
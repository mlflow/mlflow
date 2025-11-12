import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { getOffsets, genSkeletonAnimatedColor } from './utils';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingState } from '../LoadingState/LoadingState';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const paragraphContainerStyles = css({
    cursor: 'progress',
    width: '100%',
    height: 20,
    display: 'flex',
    justifyContent: 'flex-start',
    alignItems: 'center',
});
const paragraphFillStyles = css({
    borderRadius: 'var(--border-radius)',
    height: 8,
});
export const ParagraphSkeleton = ({ label, seed = '', frameRate = 60, style, loading = true, loadingDescription = 'ParagraphSkeleton', ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    const offsetWidth = getOffsets(seed)[0];
    return (_jsxs("div", { ...addDebugOutlineIfEnabled(), css: paragraphContainerStyles, style: {
            ...style,
            ['--border-radius']: `${theme.general.borderRadiusBase}px`,
        }, ...restProps, children: [loading && _jsx(LoadingState, { description: loadingDescription }), _jsx("span", { css: visuallyHidden, children: label }), _jsx("div", { "aria-hidden": true, css: [
                    paragraphFillStyles,
                    genSkeletonAnimatedColor(theme, frameRate),
                    { width: `calc(100% - ${offsetWidth}px)` },
                ] })] }));
};
//# sourceMappingURL=ParagraphSkeleton.js.map
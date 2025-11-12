import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Breadcrumb as AntDBreadcrumb } from 'antd';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronRightIcon } from '../Icon';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const Breadcrumb = /* #__PURE__ */ (() => {
    const Breadcrumb = ({ dangerouslySetAntdProps, includeTrailingCaret = true, ...props }) => {
        const { theme, classNamePrefix } = useDesignSystemTheme();
        const separatorClass = `.${classNamePrefix}-breadcrumb-separator`;
        const styles = css({
            // `antd` forces the last anchor to be black, so that it doesn't look like an anchor
            // (even though it is one). This undoes that; if the user wants to make the last
            // text-colored, they can do that by not using an anchor.
            'span:last-child a': {
                color: theme.colors.primary,
                // TODO: Need to pull a global color for anchor hover/focus. Discuss with Ginny.
                ':hover, :focus': {
                    color: '#2272B4',
                },
            },
            // TODO: Consider making this global within dubois components
            a: {
                '&:focus-visible': {
                    outlineColor: `${theme.colors.actionDefaultBorderFocus} !important`,
                    outlineStyle: 'auto !important',
                },
            },
            [separatorClass]: {
                fontSize: theme.general.iconFontSize,
            },
            '& > span': {
                display: 'inline-flex',
                alignItems: 'center',
            },
        });
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsxs(AntDBreadcrumb, { ...addDebugOutlineIfEnabled(), separator: _jsx(ChevronRightIcon, {}), ...props, ...dangerouslySetAntdProps, css: css(getAnimationCss(theme.options.enableAnimation), styles), children: [props.children, includeTrailingCaret && props.children && _jsx(Breadcrumb.Item, { children: " " })] }) }));
    };
    Breadcrumb.Item = AntDBreadcrumb.Item;
    Breadcrumb.Separator = AntDBreadcrumb.Separator;
    return Breadcrumb;
})();
//# sourceMappingURL=Breadcrumb.js.map
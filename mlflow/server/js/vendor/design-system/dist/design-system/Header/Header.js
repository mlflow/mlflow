import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import React from 'react';
import { useDesignSystemTheme } from '../Hooks';
import { Space } from '../Space';
import { Title } from '../Typography/Title';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getHeaderStyles = (clsPrefix, theme) => {
    const breadcrumbClass = `.${clsPrefix}-breadcrumb`;
    const styles = {
        [breadcrumbClass]: {
            lineHeight: theme.typography.lineHeightBase,
        },
    };
    return css(importantify(styles));
};
export const Header = ({ breadcrumbs, title, titleAddOns, dangerouslyAppendEmotionCSS, buttons, children, titleElementLevel, ...rest }) => {
    const { classNamePrefix: clsPrefix, theme } = useDesignSystemTheme();
    const buttonsArray = Array.isArray(buttons) ? buttons : buttons ? [buttons] : [];
    // TODO: Move to getHeaderStyles for consistency, followup ticket: https://databricks.atlassian.net/browse/FEINF-1222
    const styles = {
        titleWrapper: css({
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            rowGap: theme.spacing.sm,
            // Buttons have 32px height while Title level 2 elements used by this component have a height of 28px
            // These paddings enforce height to be the same without buttons too
            ...(buttonsArray.length === 0 && {
                paddingTop: breadcrumbs ? 0 : theme.spacing.xs / 2,
                paddingBottom: theme.spacing.xs / 2,
            }),
        }),
        breadcrumbWrapper: css({
            lineHeight: theme.typography.lineHeightBase,
            marginBottom: theme.spacing.xs,
        }),
        title: css({
            marginTop: 0,
            marginBottom: '0 !important',
            alignSelf: 'stretch',
        }),
        // TODO: Look into a more emotion-idomatic way of doing this.
        titleIfOtherElementsPresent: css({
            marginTop: 2,
        }),
        buttonContainer: css({
            marginLeft: 8,
        }),
        titleAddOnsWrapper: css({
            display: 'inline-flex',
            verticalAlign: 'middle',
            alignItems: 'center',
            flexWrap: 'wrap',
            marginLeft: theme.spacing.sm,
            gap: theme.spacing.xs,
        }),
    };
    return (_jsxs("div", { ...addDebugOutlineIfEnabled(), css: [getHeaderStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS], ...rest, children: [breadcrumbs && _jsx("div", { css: styles.breadcrumbWrapper, children: breadcrumbs }), _jsxs("div", { css: styles.titleWrapper, children: [_jsxs(Title, { level: 2, elementLevel: titleElementLevel, css: [styles.title, (buttons || breadcrumbs) && styles.titleIfOtherElementsPresent], children: [title, titleAddOns && _jsx("span", { css: styles.titleAddOnsWrapper, children: titleAddOns })] }), buttons && (_jsx("div", { css: styles.buttonContainer, children: _jsx(Space, { dangerouslySetAntdProps: { wrap: true }, size: 8, children: buttonsArray.filter(Boolean).map((button, i) => {
                                const defaultKey = `dubois-header-button-${i}`;
                                return React.isValidElement(button) ? (React.cloneElement(button, {
                                    key: button.key || defaultKey,
                                })) : (_jsx(React.Fragment, { children: button }, defaultKey));
                            }) }) }))] })] }));
};
//# sourceMappingURL=Header.js.map
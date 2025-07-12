import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Tabs as AntDTabs } from 'antd';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon, PlusIcon } from '../Icon';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const getLegacyTabEmotionStyles = (clsPrefix, theme) => {
    const classTab = `.${clsPrefix}-tabs-tab`;
    const classButton = `.${clsPrefix}-tabs-tab-btn`;
    const classActive = `.${clsPrefix}-tabs-tab-active`;
    const classDisabled = `.${clsPrefix}-tabs-tab-disabled`;
    const classUnderline = `.${clsPrefix}-tabs-ink-bar`;
    const classClosable = `.${clsPrefix}-tabs-tab-with-remove`;
    const classNav = `.${clsPrefix}-tabs-nav`;
    const classCloseButton = `.${clsPrefix}-tabs-tab-remove`;
    const classAddButton = `.${clsPrefix}-tabs-nav-add`;
    const styles = {
        '&&': {
            overflow: 'unset',
        },
        [classTab]: {
            borderBottom: 'none',
            backgroundColor: 'transparent',
            border: 'none',
            paddingLeft: 0,
            paddingRight: 0,
            paddingTop: 6,
            paddingBottom: 6,
            marginRight: 24,
        },
        [classButton]: {
            color: theme.colors.textSecondary,
            fontWeight: theme.typography.typographyBoldFontWeight,
            textShadow: 'none',
            fontSize: theme.typography.fontSizeMd,
            lineHeight: theme.typography.lineHeightBase,
            '&:hover': {
                color: theme.colors.actionDefaultTextHover,
            },
            '&:active': {
                color: theme.colors.actionDefaultTextPress,
            },
            outlineWidth: 2,
            outlineStyle: 'none',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineOffset: 2,
            '&:focus-visible': {
                outlineStyle: 'auto',
            },
        },
        [classActive]: {
            [classButton]: {
                color: theme.colors.textPrimary,
            },
            // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
            // jumping when switching tabs.
            boxShadow: `inset 0 -3px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,
        },
        [classDisabled]: {
            [classButton]: {
                color: theme.colors.actionDisabledText,
                '&:hover': {
                    color: theme.colors.actionDisabledText,
                },
                '&:active': {
                    color: theme.colors.actionDisabledText,
                },
            },
        },
        [classUnderline]: {
            display: 'none',
        },
        [classClosable]: {
            borderTop: 'none',
            borderLeft: 'none',
            borderRight: 'none',
            background: 'none',
            paddingTop: 0,
            paddingBottom: 0,
            height: theme.general.heightSm,
        },
        [classNav]: {
            height: theme.general.heightSm,
            '&::before': {
                borderColor: theme.colors.borderDecorative,
            },
        },
        [classCloseButton]: {
            height: 24,
            width: 24,
            padding: 6,
            borderRadius: theme.legacyBorders.borderRadiusMd,
            marginTop: 0,
            marginRight: 0,
            marginBottom: 0,
            marginLeft: 4,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.actionDefaultTextHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress,
            },
            '&:focus-visible': {
                outlineWidth: 2,
                outlineStyle: 'solid',
                outlineColor: theme.colors.actionDefaultBorderFocus,
            },
        },
        [classAddButton]: {
            backgroundColor: 'transparent',
            color: theme.colors.textValidationInfo,
            border: 'none',
            borderRadius: theme.legacyBorders.borderRadiusMd,
            margin: 4,
            height: 24,
            width: 24,
            padding: 0,
            minWidth: 'auto',
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.actionDefaultTextHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress,
            },
            '&:focus-visible': {
                outlineWidth: 2,
                outlineStyle: 'solid',
                outlineColor: theme.colors.actionDefaultBorderFocus,
            },
            '& > .anticon': {
                fontSize: 16,
            },
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    const importantStyles = importantify(styles);
    return importantStyles;
};
/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export const LegacyTabPane = ({ children, ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTabs.TabPane, { closeIcon: _jsx(CloseIcon, { css: { fontSize: theme.general.iconSize } }), ...props, ...props.dangerouslySetAntdProps, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
};
/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */
export const LegacyTabs = /* #__PURE__ */ (() => {
    const LegacyTabs = ({ editable = false, activeKey, defaultActiveKey, onChange, onEdit, children, destroyInactiveTabPane = false, dangerouslySetAntdProps = {}, dangerouslyAppendEmotionCSS = {}, ...props }) => {
        const { theme, classNamePrefix } = useDesignSystemTheme();
        return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTabs, { ...addDebugOutlineIfEnabled(), activeKey: activeKey, defaultActiveKey: defaultActiveKey, onChange: onChange, onEdit: onEdit, destroyInactiveTabPane: destroyInactiveTabPane, type: editable ? 'editable-card' : 'card', addIcon: _jsx(PlusIcon, { css: { fontSize: theme.general.iconSize } }), css: [getLegacyTabEmotionStyles(classNamePrefix, theme), importantify(dangerouslyAppendEmotionCSS)], ...dangerouslySetAntdProps, ...props, children: children }) }));
    };
    LegacyTabs.TabPane = LegacyTabPane;
    return LegacyTabs;
})();
//# sourceMappingURL=index.js.map
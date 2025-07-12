import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useCallback, useState } from 'react';
import { DropdownButton } from './Dropdown/DropdownButton';
import { Button } from '../Button';
import { getDefaultStyles, getDisabledPrimarySplitButtonStyles, getDisabledSplitButtonStyles, getPrimaryStyles, } from '../Button/styles';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronDownIcon } from '../Icon';
import { useDesignSystemSafexFlags } from '../utils';
import { importantify } from '../utils/css-utils';
const BUTTON_HORIZONTAL_PADDING = 12;
function getSplitButtonEmotionStyles(classNamePrefix, theme, useNewShadows, useNewBorderRadii, size) {
    const classDefault = `.${classNamePrefix}-btn`;
    const classPrimary = `.${classNamePrefix}-btn-primary`;
    const classDropdownTrigger = `.${classNamePrefix}-dropdown-trigger`;
    const classSmall = `.${classNamePrefix}-btn-group-sm`;
    const styles = {
        [classDefault]: {
            ...getDefaultStyles(theme),
            boxShadow: useNewShadows ? theme.shadows.xs : 'none',
            height: size === 'small' ? theme.general.iconSize : theme.general.heightSm,
            padding: `4px ${BUTTON_HORIZONTAL_PADDING}px`,
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '-2px',
                outlineColor: theme.colors.actionDefaultBorderFocus,
            },
            '.anticon, &:focus-visible .anticon': {
                color: theme.colors.textSecondary,
            },
            '&:hover .anticon': {
                color: theme.colors.actionDefaultIconHover,
            },
            '&:active .anticon': {
                color: theme.colors.actionDefaultIconPress,
            },
        },
        ...(useNewBorderRadii && {
            [`${classDefault}:first-of-type`]: {
                borderTopRightRadius: '0px !important',
                borderBottomRightRadius: '0px !important',
            },
        }),
        [classPrimary]: {
            ...getPrimaryStyles(theme),
            ...(useNewShadows && {
                boxShadow: theme.shadows.xs,
            }),
            [`&:first-of-type`]: {
                borderRight: `1px solid ${theme.colors.actionPrimaryTextDefault}`,
                marginRight: 1,
            },
            [classDropdownTrigger]: {
                borderLeft: `1px solid ${theme.colors.actionPrimaryTextDefault}`,
            },
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: '1px',
                outlineOffset: '-3px',
                outlineColor: theme.colors.white,
            },
            '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
                color: theme.colors.actionPrimaryIcon,
            },
        },
        [classDropdownTrigger]: {
            // Needs to be 1px less than our standard 8px to allow for the off-by-one border handling in this component.
            padding: 3,
            borderLeftColor: 'transparent',
            width: theme.general.heightSm,
        },
        [`&${classSmall}`]: {
            [classDropdownTrigger]: {
                padding: 5,
            },
        },
        '&&': {
            [`[disabled], ${classPrimary}[disabled]`]: {
                ...getDisabledSplitButtonStyles(theme, useNewShadows),
                ...(useNewShadows && {
                    boxShadow: 'none',
                }),
                [`&:first-of-type`]: {
                    borderRight: `1px solid ${theme.colors.actionPrimaryIcon}`,
                    marginRight: 1,
                },
                [classDropdownTrigger]: {
                    borderLeft: `1px solid ${theme.colors.actionPrimaryIcon}`,
                },
                '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
                    color: theme.colors.actionDisabledText,
                },
            },
            [`${classPrimary}[disabled]`]: {
                ...getDisabledPrimarySplitButtonStyles(theme, useNewShadows),
                '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
                    color: theme.colors.actionPrimaryTextDefault,
                },
            },
        },
        [`${classDefault}:not(:first-of-type)`]: {
            width: theme.general.heightSm,
            padding: '3px !important',
            ...(useNewBorderRadii && {
                borderTopLeftRadius: '0px !important',
                borderBottomLeftRadius: '0px !important',
            }),
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    const importantStyles = importantify(styles);
    return css(importantStyles);
}
export const SplitButton = (props) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
    const { children, icon, deprecatedMenu, type, loading, loadingButtonStyles, placement, dangerouslySetAntdProps, size, ...dropdownButtonProps } = props;
    // Size of button when loading only icon is shown
    const LOADING_BUTTON_SIZE = theme.general.iconFontSize + 2 * BUTTON_HORIZONTAL_PADDING + 2 * theme.general.borderWidth;
    const [width, setWidth] = useState(LOADING_BUTTON_SIZE);
    // Set the width to the button's width in regular state to later use when in loading state
    // We do this to have just a loading icon in loading state at the normal width to avoid flicker and width changes in page
    const ref = useCallback((node) => {
        if (node && !loading) {
            setWidth(node.getBoundingClientRect().width);
        }
    }, [loading]);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx("div", { ref: ref, css: { display: 'inline-flex', position: 'relative', verticalAlign: 'middle' }, children: loading ? (_jsx(Button, { componentId: "codegen_design-system_src_design-system_splitbutton_splitbutton.tsx_163", type: type === 'default' ? undefined : type, style: {
                    width: width,
                    fontSize: theme.general.iconFontSize,
                    ...loadingButtonStyles,
                }, loading: true, htmlType: props.htmlType, title: props.title, className: props.className, children: children })) : (_jsx(DropdownButton, { ...dropdownButtonProps, overlay: deprecatedMenu, trigger: ['click'], css: getSplitButtonEmotionStyles(classNamePrefix, theme, useNewShadows, useNewBorderRadii, size), icon: _jsx(ChevronDownIcon, { css: { fontSize: theme.general.iconFontSize }, "aria-hidden": "true" }), placement: placement || 'bottomRight', type: type === 'default' ? undefined : type, leftButtonIcon: icon, ...dangerouslySetAntdProps, children: children })) }) }));
};
//# sourceMappingURL=SplitButton.js.map
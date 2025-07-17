import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { ClassNames, css } from '@emotion/react';
import { Select as AntDSelect } from 'antd';
import _ from 'lodash';
import { forwardRef, useEffect, useState } from 'react';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { CheckIcon, ChevronDownIcon, CloseIcon, LoadingIcon, XCircleFillIcon } from '../Icon';
import { LoadingState } from '../LoadingState/LoadingState';
import { useDesignSystemSafexFlags } from '../utils';
import { getDarkModePortalStyles, getValidationStateColor, importantify } from '../utils/css-utils';
import { addDebugOutlineStylesIfEnabled } from '../utils/debug';
function getSelectEmotionStyles({ clsPrefix, theme, validationState, useNewFormUISpacing, }) {
    const classFocused = `.${clsPrefix}-focused`;
    const classOpen = `.${clsPrefix}-open`;
    const classSingle = `.${clsPrefix}-single`;
    const classSelector = `.${clsPrefix}-selector`;
    const classDisabled = `.${clsPrefix}-disabled`;
    const classMultiple = `.${clsPrefix}-multiple`;
    const classItem = `.${clsPrefix}-selection-item`;
    const classItemOverflowContainer = `.${clsPrefix}-selection-overflow`;
    const classItemOverflowItem = `.${clsPrefix}-selection-overflow-item`;
    const classItemOverflowSuffix = `.${clsPrefix}-selection-overflow-item-suffix`;
    const classArrow = `.${clsPrefix}-arrow`;
    const classArrowLoading = `.${clsPrefix}-arrow-loading`;
    const classPlaceholder = `.${clsPrefix}-selection-placeholder`;
    const classCloseButton = `.${clsPrefix}-selection-item-remove`;
    const classSearch = `.${clsPrefix}-selection-search`;
    const classShowSearch = `.${clsPrefix}-show-search`;
    const classSearchClear = `.${clsPrefix}-clear`;
    const classAllowClear = `.${clsPrefix}-allow-clear`;
    const classSearchInput = `.${clsPrefix}-selection-search-input`;
    const classFormMessage = `.${clsPrefix.replace('-select', '')}-form-message`;
    const validationColor = getValidationStateColor(theme, validationState);
    const styles = {
        ...addDebugOutlineStylesIfEnabled(theme),
        ...(useNewFormUISpacing && {
            [`& + ${classFormMessage}`]: {
                marginTop: theme.spacing.sm,
            },
        }),
        '&:hover': {
            [classSelector]: {
                borderColor: theme.colors.actionDefaultBorderHover,
            },
        },
        [classSelector]: {
            paddingLeft: 12,
            // Only the select _item_ is clickable, so we need to have zero padding here, and add it on the item itself,
            // to make sure the whole select is clickable.
            paddingRight: 0,
            color: theme.colors.textPrimary,
            backgroundColor: 'transparent',
            height: theme.general.heightSm,
            '::after': {
                lineHeight: theme.typography.lineHeightBase,
            },
            '::before': {
                lineHeight: theme.typography.lineHeightBase,
            },
        },
        [classSingle]: {
            [`&${classSelector}`]: {
                height: theme.general.heightSm,
            },
        },
        [classItem]: {
            color: theme.colors.textPrimary,
            paddingRight: 32,
            lineHeight: theme.typography.lineHeightBase,
            paddingTop: 5,
            paddingBottom: 5,
        },
        // Note: This supports search, which we don't support. The styles here support legacy usages.
        [classSearch]: {
            right: 24,
            left: 8,
            marginInlineStart: 4,
            [classSearchInput]: {
                color: theme.colors.actionDefaultTextDefault,
                height: 24,
            },
        },
        [`&${classSingle}`]: {
            [classSearchInput]: {
                height: theme.general.heightSm,
            },
        },
        // Note: This supports search, which we don't support. The styles here support legacy usages.
        [`&${classShowSearch}${classOpen}${classSingle}`]: {
            [classItem]: {
                color: theme.colors.actionDisabledText,
            },
        },
        // Note: This supports search, which we don't support. The styles here support legacy usages.
        [classSearchClear]: {
            right: 24,
            backgroundColor: 'transparent',
        },
        [`&${classFocused}`]: {
            [classSelector]: {
                outlineColor: theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                borderColor: 'transparent',
                boxShadow: 'none',
            },
        },
        [`&${classDisabled}`]: {
            [classSelector]: {
                backgroundColor: theme.colors.actionDisabledBackground,
                color: theme.colors.actionDisabledText,
                border: 'transparent',
            },
            [classItem]: {
                color: theme.colors.actionDisabledText,
            },
            [classArrow]: {
                color: theme.colors.actionDisabledText,
            },
        },
        [classArrow]: {
            height: theme.general.iconFontSize,
            width: theme.general.iconFontSize,
            top: (theme.general.heightSm - theme.general.iconFontSize) / 2,
            marginTop: 0,
            color: theme.colors.textSecondary,
            fontSize: theme.general.iconFontSize,
            '.anticon': {
                // For some reason ant sets this to 'auto'. Need to set it back to 'none' to allow the element below to receive
                // the click event.
                pointerEvents: 'none',
                // anticon default line height is 0 and that wrongly shifts the icon down
                lineHeight: 1,
            },
            [`&${classArrowLoading}`]: {
                top: (theme.general.heightSm - theme.general.iconFontSize) / 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: theme.general.iconFontSize,
            },
        },
        [classPlaceholder]: {
            color: theme.colors.textPlaceholder,
            right: 'auto',
            left: 'auto',
            width: '100%',
            paddingRight: 32,
            lineHeight: theme.typography.lineHeightBase,
            alignSelf: 'center',
        },
        [`&${classMultiple}`]: {
            [classSelector]: {
                paddingTop: 3,
                paddingBottom: 3,
                paddingLeft: 8,
                paddingRight: 30,
                minHeight: theme.general.heightSm,
                height: 'auto',
                '&::after': {
                    margin: 0,
                },
            },
            [classItem]: {
                backgroundColor: theme.colors.tagDefault,
                color: theme.colors.textPrimary,
                border: 'none',
                height: 20,
                lineHeight: theme.typography.lineHeightBase,
                fontSize: theme.typography.fontSizeBase,
                marginInlineEnd: 4,
                marginTop: 2,
                marginBottom: 2,
                paddingRight: 0,
                paddingTop: 0,
                paddingBottom: 0,
            },
            [classItemOverflowContainer]: {
                minHeight: 24,
            },
            [classItemOverflowItem]: {
                alignSelf: 'auto',
                height: 24,
                lineHeight: theme.typography.lineHeightBase,
            },
            [classSearch]: {
                marginTop: 0,
                left: 0,
                right: 0,
            },
            [`&${classDisabled}`]: {
                [classItem]: {
                    paddingRight: 2,
                },
            },
            [classArrow]: {
                top: (theme.general.heightSm - theme.general.iconFontSize) / 2,
            },
            [`&${classAllowClear}`]: {
                [classSearchClear]: {
                    top: (theme.general.heightSm - theme.general.iconFontSize + 4) / 2,
                },
            },
            [classPlaceholder]: {
                // Compensate for the caret placeholder width
                paddingLeft: 4,
                color: theme.colors.textPlaceholder,
            },
            [`&:not(${classFocused})`]: {
                [classItemOverflowSuffix]: {
                    // Do not keep the caret's placeholder at full height when not focused,
                    // because it introduces a new line even when not focused. Using display: none would break the caret
                    height: 0,
                },
            },
        },
        [`&${classMultiple}${classDisabled}`]: {
            [classItem]: {
                color: theme.colors.actionDisabledText,
            },
        },
        [`&${classAllowClear}`]: {
            [classItem]: {
                paddingRight: 0,
            },
            [classSelector]: {
                paddingRight: 52,
            },
            [classSearchClear]: {
                top: (theme.general.heightSm - theme.general.iconFontSize + 4) / 2,
                opacity: 100,
                width: theme.general.iconFontSize,
                height: theme.general.iconFontSize,
                marginTop: 0,
            },
        },
        [classCloseButton]: {
            color: theme.colors.textPrimary,
            borderTopRightRadius: theme.legacyBorders.borderRadiusMd,
            borderBottomRightRadius: theme.legacyBorders.borderRadiusMd,
            height: theme.general.iconFontSize,
            width: theme.general.iconFontSize,
            lineHeight: theme.typography.lineHeightBase,
            paddingInlineEnd: 0,
            marginInlineEnd: 0,
            '& > .anticon': {
                height: theme.general.iconFontSize - 4,
                fontSize: theme.general.iconFontSize - 4,
            },
            '&:hover': {
                color: theme.colors.actionTertiaryTextHover,
                backgroundColor: theme.colors.tagHover,
            },
            '&:active': {
                color: theme.colors.actionTertiaryTextPress,
                backgroundColor: theme.colors.tagPress,
            },
        },
        ...(validationState && {
            [`& > ${classSelector}`]: {
                borderColor: validationColor,
                '&:hover': {
                    borderColor: validationColor,
                },
            },
            [`&${classFocused} > ${classSelector}`]: {
                outlineColor: validationColor,
                outlineOffset: -2,
            },
        }),
        ...getAnimationCss(theme.options.enableAnimation),
    };
    const importantStyles = importantify(styles);
    return css(importantStyles);
}
function getDropdownStyles(clsPrefix, theme, useNewShadows, useNewBorderColors) {
    const classItem = `.${clsPrefix}-item-option`;
    const classItemActive = `.${clsPrefix}-item-option-active`;
    const classItemSelected = `.${clsPrefix}-item-option-selected`;
    const classItemState = `.${clsPrefix}-item-option-state`;
    const styles = {
        borderColor: useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative,
        borderWidth: 1,
        borderStyle: 'solid',
        zIndex: theme.options.zIndexBase + 50,
        boxShadow: theme.general.shadowLow,
        ...addDebugOutlineStylesIfEnabled(theme),
        [classItem]: {
            height: theme.general.heightSm,
        },
        [classItemActive]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            height: theme.general.heightSm,
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            },
        },
        [classItemSelected]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            fontWeight: 'normal',
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            },
        },
        [classItemState]: {
            color: theme.colors.textSecondary,
            '& > span': {
                verticalAlign: 'middle',
            },
        },
        [`.${clsPrefix}-loading-options`]: {
            pointerEvents: 'none',
            margin: '0 auto',
            height: theme.general.heightSm,
            display: 'block',
        },
        ...getAnimationCss(theme.options.enableAnimation),
        ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
    };
    const importantStyles = importantify(styles);
    return css(importantStyles);
}
function getLoadingIconStyles(theme) {
    return css({ fontSize: 20, color: theme.colors.textSecondary, lineHeight: '20px' });
}
const scrollbarVisibleItemsCount = 8;
const getIconSizeStyle = (theme, newIconDefault) => importantify({
    fontSize: newIconDefault ?? theme.general.iconFontSize,
});
function DuboisSelect({ children, validationState, loading, loadingDescription = 'Select', mode, options, notFoundContent, optionFilterProp, dangerouslySetAntdProps, virtual, dropdownClassName, id, onDropdownVisibleChange, maxHeight, ...restProps }, ref) {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const { useNewShadows, useNewFormUISpacing, useNewBorderColors } = useDesignSystemSafexFlags();
    const clsPrefix = getPrefixedClassName('select');
    const [isOpen, setIsOpen] = useState(false);
    const [uniqueId, setUniqueId] = useState('');
    // Antd's default is 256, to show half an extra item when scrolling we add 0.5 height extra
    // Reducing to 5.5 as it's default with other components is not an option here because it would break existing usages relying on 8 items being shown by default
    const MAX_HEIGHT = maxHeight ?? theme.general.heightSm * 8.5;
    useEffect(() => {
        setUniqueId(id || _.uniqueId('dubois-select-'));
    }, [id]);
    useEffect(() => {
        // Ant doesn't populate aria-expanded on init (only on user interaction) so we need to do it ourselves
        // in order to pass accessibility tests (Microsoft Accessibility Insights). See: JOBS-11125
        document.getElementById(uniqueId)?.setAttribute('aria-expanded', 'false');
    }, [uniqueId]);
    return (_jsx(ClassNames, { children: ({ css }) => {
            return (_jsxs(DesignSystemAntDConfigProvider, { children: [loading && _jsx(LoadingState, { description: loadingDescription }), _jsx(AntDSelect, { onDropdownVisibleChange: (visible) => {
                            onDropdownVisibleChange?.(visible);
                            setIsOpen(visible);
                        }, ...(!isOpen
                            ? {
                                'aria-owns': undefined,
                                'aria-controls': undefined,
                                'aria-activedescendant': undefined,
                            }
                            : {}), id: uniqueId, css: getSelectEmotionStyles({ clsPrefix, theme, validationState, useNewFormUISpacing }), removeIcon: _jsx(CloseIcon, { "aria-hidden": "false", css: getIconSizeStyle(theme) }), clearIcon: _jsx(XCircleFillIcon, { "aria-hidden": "false", css: getIconSizeStyle(theme, 12), "aria-label": "close-circle" }), ref: ref, suffixIcon: loading && mode === 'tags' ? (_jsx(LoadingIcon, { spin: true, "aria-label": "loading", "aria-hidden": "false", css: getIconSizeStyle(theme, 12) })) : (_jsx(ChevronDownIcon, { css: getIconSizeStyle(theme) })), menuItemSelectedIcon: _jsx(CheckIcon, { css: getIconSizeStyle(theme) }), showArrow: true, dropdownMatchSelectWidth: true, notFoundContent: notFoundContent ?? (_jsx("div", { css: { color: theme.colors.textSecondary, textAlign: 'center' }, children: "No results found" })), dropdownClassName: css([
                            getDropdownStyles(clsPrefix, theme, useNewShadows, useNewBorderColors),
                            dropdownClassName,
                        ]), listHeight: MAX_HEIGHT, maxTagPlaceholder: (items) => `+ ${items.length} more`, mode: mode, options: options, loading: loading, filterOption: true, 
                        // NOTE(FEINF-1102): This is needed to avoid ghost scrollbar that generates error when clicked on exactly 8 elements
                        // Because by default AntD uses true for virtual, we want to replicate the same even if there are no children
                        virtual: virtual ??
                            ((children && Array.isArray(children) && children.length !== scrollbarVisibleItemsCount) ||
                                (options && options.length !== scrollbarVisibleItemsCount) ||
                                (!children && !options)), optionFilterProp: optionFilterProp ?? 'children', ...restProps, ...dangerouslySetAntdProps, children: loading && mode !== 'tags' ? (_jsxs(_Fragment, { children: [children, _jsx(LegacyOption, { disabled: true, value: "select-loading-options", className: `${clsPrefix}-loading-options`, children: _jsx(LoadingIcon, { "aria-hidden": "false", spin: true, css: getLoadingIconStyles(theme), "aria-label": "loading" }) })] })) : (children) })] }));
        } }));
}
export const LegacySelectOption = forwardRef(function Option(props, ref) {
    const { dangerouslySetAntdProps, ...restProps } = props;
    return _jsx(AntDSelect.Option, { ...restProps, ref: ref, ...dangerouslySetAntdProps });
});
// Needed for rc-select to not throw warning about our component not being Select.Option
LegacySelectOption.isSelectOption = true;
/**
 * @deprecated use LegacySelect.Option instead
 */
export const LegacyOption = LegacySelectOption;
export const LegacySelectOptGroup = /* #__PURE__ */ (() => {
    const OptGroup = forwardRef(function OptGroup(props, ref) {
        return _jsx(AntDSelect.OptGroup, { ...props, isSelectOptGroup: true, ref: ref });
    });
    // Needed for antd to work properly and for rc-select to not throw warning about our component not being Select.OptGroup
    OptGroup.isSelectOptGroup = true;
    return OptGroup;
})();
/**
 * @deprecated use LegacySelect.OptGroup instead
 */
export const LegacyOptGroup = LegacySelectOptGroup;
/**
 * @deprecated Use Select, TypeaheadCombobox, or DialogCombobox depending on your use-case. See http://go/deprecate-ant-select for more information
 */
export const LegacySelect = /* #__PURE__ */ (() => {
    const DuboisRefForwardedSelect = forwardRef(DuboisSelect);
    DuboisRefForwardedSelect.Option = LegacySelectOption;
    DuboisRefForwardedSelect.OptGroup = LegacySelectOptGroup;
    return DuboisRefForwardedSelect;
})();
//# sourceMappingURL=LegacySelect.js.map
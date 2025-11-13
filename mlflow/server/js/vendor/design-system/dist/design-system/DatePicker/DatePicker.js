import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { DatePicker as AntDDatePicker } from 'antd';
import { forwardRef, useEffect, useRef } from 'react';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { addDebugOutlineIfEnabled } from '../utils/debug';
function getEmotionStyles(clsPrefix, theme) {
    const classFocused = `.${clsPrefix}-focused`;
    const classActiveBar = `.${clsPrefix}-active-bar`;
    const classSeparator = `.${clsPrefix}-separator`;
    const classSuffix = `.${clsPrefix}-suffix`;
    const styles = {
        height: 32,
        borderRadius: theme.borders.borderRadiusSm,
        borderColor: theme.colors.border,
        color: theme.colors.textPrimary,
        transition: 'border 0s, box-shadow 0s',
        [`&${classFocused},:hover`]: {
            borderColor: theme.colors.actionDefaultBorderHover,
        },
        '&:active': {
            borderColor: theme.colors.actionDefaultBorderPress,
        },
        [`&${classFocused}`]: {
            boxShadow: `none !important`,
            outline: `${theme.colors.actionDefaultBorderFocus} solid 2px !important`,
            outlineOffset: '-2px !important',
            borderColor: 'transparent !important',
        },
        [`& ${classActiveBar}`]: {
            background: `${theme.colors.actionDefaultBorderPress} !important`,
        },
        [`& input::placeholder, & ${classSeparator}, & ${classSuffix}`]: {
            color: theme.colors.textPrimary,
        },
    };
    return css(styles);
}
const getDropdownStyles = (theme) => {
    return {
        zIndex: theme.options.zIndexBase + 50,
    };
};
function useDatePickerStyles() {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const clsPrefix = getPrefixedClassName('picker');
    return getEmotionStyles(clsPrefix, theme);
}
const AccessibilityWrapper = ({ children, ariaLive = 'assertive', ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    useEffect(() => {
        if (ref.current) {
            const inputs = theme.isDarkMode
                ? ref.current.querySelectorAll('.du-bois-dark-picker-input > input')
                : ref.current.querySelectorAll('.du-bois-light-picker-input > input');
            inputs.forEach((input) => input.setAttribute('aria-live', ariaLive));
        }
    }, [ref, ariaLive, theme.isDarkMode]);
    return (_jsx("div", { ...restProps, ref: ref, children: children }));
};
export const DuboisDatePicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...addDebugOutlineIfEnabled(), ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker, { css: styles, ref: ref, ...restProps, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
const RangePicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker.RangePicker, { ...addDebugOutlineIfEnabled(), css: styles, ...restProps, ref: ref, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
const TimePicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...addDebugOutlineIfEnabled(), ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker.TimePicker, { css: styles, ...restProps, ref: ref, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
const QuarterPicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...addDebugOutlineIfEnabled(), ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker.QuarterPicker, { css: styles, ...restProps, ref: ref, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
const WeekPicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...addDebugOutlineIfEnabled(), ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker.WeekPicker, { css: styles, ...restProps, ref: ref, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
const MonthPicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...addDebugOutlineIfEnabled(), ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker.MonthPicker, { css: styles, ...restProps, ref: ref, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
const YearPicker = forwardRef((props, ref) => {
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AccessibilityWrapper, { ...addDebugOutlineIfEnabled(), ...wrapperDivProps, ariaLive: ariaLive, children: _jsx(AntDDatePicker.YearPicker, { css: styles, ...restProps, ref: ref, popupStyle: { ...getDropdownStyles(theme), ...(props.popupStyle || {}) } }) }) }));
});
/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */
export const LegacyDatePicker = /* #__PURE__ */ Object.assign(DuboisDatePicker, {
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    RangePicker,
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    TimePicker,
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    QuarterPicker,
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    WeekPicker,
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    MonthPicker,
    /**
     * See deprecation notice for `LegacyDatePicker`.
     * @deprecated
     */
    YearPicker,
});
//# sourceMappingURL=DatePicker.js.map
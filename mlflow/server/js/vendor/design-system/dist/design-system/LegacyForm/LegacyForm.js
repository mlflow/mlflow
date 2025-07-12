import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Form as AntDForm } from 'antd';
import { forwardRef } from 'react';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getFormItemEmotionStyles = ({ theme, clsPrefix }) => {
    const clsFormItemLabel = `.${clsPrefix}-form-item-label`;
    const clsFormItemInputControl = `.${clsPrefix}-form-item-control-input`;
    const clsFormItemExplain = `.${clsPrefix}-form-item-explain`;
    const clsHasError = `.${clsPrefix}-form-item-has-error`;
    return css({
        [clsFormItemLabel]: {
            fontWeight: theme.typography.typographyBoldFontWeight,
            lineHeight: theme.typography.lineHeightBase,
            '.anticon': {
                fontSize: theme.general.iconFontSize,
            },
        },
        [clsFormItemExplain]: {
            fontSize: theme.typography.fontSizeSm,
            margin: 0,
            [`&${clsFormItemExplain}-success`]: {
                color: theme.colors.textValidationSuccess,
            },
            [`&${clsFormItemExplain}-warning`]: {
                color: theme.colors.textValidationDanger,
            },
            [`&${clsFormItemExplain}-error`]: {
                color: theme.colors.textValidationDanger,
            },
            [`&${clsFormItemExplain}-validating`]: {
                color: theme.colors.textSecondary,
            },
        },
        [clsFormItemInputControl]: {
            minHeight: theme.general.heightSm,
        },
        [`${clsFormItemInputControl} input[disabled]`]: {
            border: 'none',
        },
        [`&${clsHasError} input:focus`]: importantify({
            boxShadow: 'none',
        }),
        ...getAnimationCss(theme.options.enableAnimation),
    });
};
/**
 * @deprecated Use `Form` from `@databricks/design-system/development` instead.
 */
export const LegacyFormDubois = forwardRef(function Form({ dangerouslySetAntdProps, children, ...props }, ref) {
    const mergedProps = {
        ...props,
        layout: props.layout || 'vertical',
        requiredMark: props.requiredMark || false,
    };
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDForm, { ...addDebugOutlineIfEnabled(), ...mergedProps, colon: false, ref: ref, ...dangerouslySetAntdProps, children: _jsx(RestoreAntDDefaultClsPrefix, { children: children }) }) }));
});
const FormItem = ({ dangerouslySetAntdProps, children, ...props }) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDForm.Item, { ...addDebugOutlineIfEnabled(), ...props, css: getFormItemEmotionStyles({
                theme,
                clsPrefix: classNamePrefix,
            }), ...dangerouslySetAntdProps, children: children }) }));
};
const FormNamespace = /* #__PURE__ */ Object.assign(LegacyFormDubois, {
    Item: FormItem,
    List: AntDForm.List,
    useForm: AntDForm.useForm,
});
export const LegacyForm = FormNamespace;
// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
export const __INTERNAL_DO_NOT_USE__FormItem = FormItem;
//# sourceMappingURL=LegacyForm.js.map
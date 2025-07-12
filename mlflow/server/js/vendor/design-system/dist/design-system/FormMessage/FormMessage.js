import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CheckCircleIcon, DangerIcon, WarningIcon, InfoIcon } from '../Icon';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getMessageStyles = (clsPrefix, theme) => {
    const errorClass = `.${clsPrefix}-form-error-message`;
    const infoClass = `.${clsPrefix}-form-info-message`;
    const successClass = `.${clsPrefix}-form-success-message`;
    const warningClass = `.${clsPrefix}-form-warning-message`;
    const styles = {
        '&&': {
            lineHeight: theme.typography.lineHeightSm,
            fontSize: theme.typography.fontSizeSm,
            marginTop: theme.spacing.sm,
            display: 'flex',
            alignItems: 'start',
        },
        [`&${errorClass}`]: {
            color: theme.colors.actionDangerPrimaryBackgroundDefault,
        },
        [`&${infoClass}`]: {
            color: theme.colors.textPrimary,
        },
        [`&${successClass}`]: {
            color: theme.colors.textValidationSuccess,
        },
        [`&${warningClass}`]: {
            color: theme.colors.textValidationWarning,
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    return css(styles);
};
const VALIDATION_STATE_ICONS = {
    error: DangerIcon,
    success: CheckCircleIcon,
    warning: WarningIcon,
    info: InfoIcon,
};
export function FormMessage({ id, message, type = 'error', className = '', css }) {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const stateClass = `${classNamePrefix}-form-${type}-message`;
    const StateIcon = VALIDATION_STATE_ICONS[type];
    const wrapperClass = `${classNamePrefix}-form-message ${className} ${stateClass}`.trim();
    return (_jsxs("div", { ...(id && { id }), className: wrapperClass, ...addDebugOutlineIfEnabled(), css: [getMessageStyles(classNamePrefix, theme), css], role: "alert", children: [_jsx(StateIcon, {}), _jsx("div", { style: { paddingLeft: theme.spacing.xs }, children: message })] }));
}
//# sourceMappingURL=FormMessage.js.map
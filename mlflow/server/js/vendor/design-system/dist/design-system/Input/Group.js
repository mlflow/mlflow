import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Input as AntDInput } from 'antd';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getInputGroupStyling = (clsPrefix, theme, useNewShadows, useNewBorderRadii) => {
    const inputClass = `.${clsPrefix}-input`;
    const buttonClass = `.${clsPrefix}-btn`;
    return css({
        display: 'inline-flex !important',
        width: 'auto',
        [`& > ${inputClass}`]: {
            flexGrow: 1,
            ...(useNewBorderRadii && {
                borderTopRightRadius: '0px !important',
                borderBottomRightRadius: '0px !important',
            }),
            '&:disabled': {
                border: 'none',
                background: theme.colors.actionDisabledBackground,
                '&:hover': {
                    borderRight: `1px solid ${theme.colors.actionDisabledBorder} !important`,
                },
            },
            '&[data-validation]': {
                marginRight: 0,
            },
        },
        ...(useNewShadows && {
            [`& > ${buttonClass}`]: {
                boxShadow: 'none !important',
            },
        }),
        ...(useNewBorderRadii && {
            [`& > ${buttonClass}`]: {
                borderTopLeftRadius: '0px !important',
                borderBottomLeftRadius: '0px !important',
            },
        }),
        [`& > ${buttonClass} > span`]: {
            verticalAlign: 'middle',
        },
        [`& > ${buttonClass}:disabled, & > ${buttonClass}:disabled:hover`]: {
            borderLeft: `1px solid ${theme.colors.actionDisabledBorder} !important`,
            backgroundColor: `${theme.colors.actionDisabledBackground} !important`,
            color: `${theme.colors.actionDisabledText} !important`,
        },
    });
};
export const Group = ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, compact = true, ...props }) => {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDInput.Group, { ...addDebugOutlineIfEnabled(), css: [
                getInputGroupStyling(classNamePrefix, theme, useNewShadows, useNewBorderRadii),
                dangerouslyAppendEmotionCSS,
            ], compact: compact, ...props, ...dangerouslySetAntdProps }) }));
};
//# sourceMappingURL=Group.js.map
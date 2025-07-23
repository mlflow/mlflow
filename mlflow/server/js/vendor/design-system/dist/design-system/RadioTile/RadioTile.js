import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useDesignSystemTheme } from '../Hooks';
import { Radio, useRadioGroupContext } from '../Radio/Radio';
const getRadioTileStyles = (theme, classNamePrefix, maxWidth) => {
    const radioWrapper = `.${classNamePrefix}-radio-wrapper`;
    return css({
        '&&': {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
            padding: theme.spacing.md,
            gap: theme.spacing.xs,
            borderRadius: theme.borders.borderRadiusSm,
            background: 'transparent',
            cursor: 'pointer',
            ...(maxWidth && {
                maxWidth,
            }),
            // Label, radio and icon container
            '& > div:first-of-type': {
                width: '100%',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: theme.spacing.sm,
            },
            // Description container
            '& > div:nth-of-type(2)': {
                alignSelf: 'flex-start',
                textAlign: 'left',
                color: theme.colors.textSecondary,
                fontSize: theme.typography.fontSizeSm,
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                borderColor: theme.colors.actionDefaultBorderHover,
            },
            '&:disabled': {
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: 'transparent',
                cursor: 'not-allowed',
                '& > div:nth-of-type(2)': {
                    color: theme.colors.actionDisabledText,
                },
            },
        },
        [radioWrapper]: {
            display: 'flex',
            flexDirection: 'row-reverse',
            justifyContent: 'space-between',
            flex: 1,
            margin: 0,
            '& > span': {
                padding: 0,
            },
            '::after': {
                display: 'none',
            },
        },
    });
};
export const RadioTile = (props) => {
    const { description, icon, maxWidth, checked, defaultChecked, onChange, ...rest } = props;
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { value: groupValue, onChange: groupOnChange } = useRadioGroupContext();
    return (_jsxs("button", { role: "radio", type: "button", "aria-checked": groupValue === props.value, onClick: () => {
            if (props.disabled) {
                return;
            }
            onChange?.(props.value);
            groupOnChange?.({ target: { value: props.value } });
        }, tabIndex: 0, className: `${classNamePrefix}-radio-tile`, css: getRadioTileStyles(theme, classNamePrefix, maxWidth), disabled: props.disabled, children: [_jsxs("div", { children: [icon ? (_jsx("span", { css: { color: props.disabled ? theme.colors.actionDisabledText : theme.colors.textSecondary }, children: icon })) : null, _jsx(Radio, { __INTERNAL_DISABLE_RADIO_ROLE: true, ...rest, tabIndex: -1 })] }), description ? _jsx("div", { children: description }) : null] }));
};
//# sourceMappingURL=RadioTile.js.map
export function getDefaultStyles(theme, loading = false) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionDefaultBackgroundDefault,
        borderColor: theme.colors.actionDefaultBorderDefault,
        color: theme.colors.actionDefaultTextDefault,
    };
    return {
        ...defaultStyles,
        lineHeight: theme.typography.lineHeightBase,
        textDecoration: 'none',
        '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            color: theme.colors.actionDefaultTextHover,
        },
        '&:active': {
            backgroundColor: loading
                ? theme.colors.actionDefaultBackgroundDefault
                : theme.colors.actionDefaultBackgroundPress,
            borderColor: theme.colors.actionDefaultBorderPress,
            color: theme.colors.actionDefaultTextPress,
        },
    };
}
export function getPrimaryStyles(theme) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        borderColor: 'transparent',
        color: theme.colors.actionPrimaryTextDefault,
    };
    return {
        ...defaultStyles,
        textShadow: 'none',
        '&:hover': {
            backgroundColor: theme.colors.actionPrimaryBackgroundHover,
            borderColor: 'transparent',
            color: theme.colors.actionPrimaryTextHover,
        },
        '&:active': {
            backgroundColor: theme.colors.actionPrimaryBackgroundPress,
            borderColor: 'transparent',
            color: theme.colors.actionPrimaryTextPress,
        },
    };
}
export function getLinkStyles(theme) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
        borderColor: theme.colors.actionTertiaryBackgroundDefault,
        color: theme.colors.actionTertiaryTextDefault,
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            borderColor: 'transparent',
            color: theme.colors.actionTertiaryTextHover,
        },
        '&:active': {
            backgroundColor: theme.colors.actionTertiaryBackgroundPress,
            borderColor: 'transparent',
            color: theme.colors.actionTertiaryTextPress,
        },
        '&[disabled]:hover': {
            background: 'none',
            color: theme.colors.actionDisabledText,
        },
    };
}
export function getPrimaryDangerStyles(theme) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
        borderColor: 'transparent',
        color: theme.colors.actionPrimaryTextDefault,
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionDangerPrimaryBackgroundHover,
            borderColor: 'transparent',
            color: theme.colors.actionPrimaryTextHover,
        },
        '&:active': {
            backgroundColor: theme.colors.actionDangerPrimaryBackgroundPress,
            borderColor: 'transparent',
            color: theme.colors.actionPrimaryTextPress,
        },
        '&:focus-visible': {
            outlineColor: theme.colors.actionDangerPrimaryBackgroundDefault,
        },
    };
}
export function getSecondaryDangerStyles(theme) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionDangerDefaultBackgroundDefault,
        borderColor: theme.colors.actionDangerDefaultBorderDefault,
        color: theme.colors.actionDangerDefaultTextDefault,
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionDangerDefaultBackgroundHover,
            borderColor: theme.colors.actionDangerDefaultBorderHover,
            color: theme.colors.actionDangerDefaultTextHover,
        },
        '&:active': {
            backgroundColor: theme.colors.actionDangerDefaultBackgroundPress,
            borderColor: theme.colors.actionDangerDefaultBorderPress,
            color: theme.colors.actionDangerDefaultTextPress,
        },
        '&:focus-visible': {
            outlineColor: theme.colors.actionDangerPrimaryBackgroundDefault,
        },
    };
}
export function getDisabledDefaultStyles(theme, useNewShadows) {
    const defaultStyles = {
        backgroundColor: 'transparent',
        borderColor: theme.colors.actionDisabledBorder,
        color: theme.colors.actionDisabledText,
        ...(useNewShadows && {
            boxShadow: 'none',
        }),
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: 'transparent',
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
        },
        '&:active': {
            backgroundColor: 'transparent',
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
        },
    };
}
export function getDisabledPrimaryStyles(theme, useNewShadows) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionDisabledBorder,
        borderColor: 'transparent',
        color: theme.colors.actionPrimaryTextDefault,
        ...(useNewShadows && {
            boxShadow: 'none',
        }),
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionDisabledBorder,
            borderColor: 'transparent',
            color: theme.colors.actionPrimaryTextDefault,
        },
        '&:active': {
            backgroundColor: theme.colors.actionDisabledBorder,
            borderColor: 'transparent',
            color: theme.colors.actionPrimaryTextDefault,
        },
    };
}
export function getDisabledErrorStyles(theme, useNewShadows) {
    return getDisabledPrimaryStyles(theme, useNewShadows);
}
export function getDisabledTertiaryStyles(theme, useNewShadows) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
        borderColor: 'transparent',
        color: theme.colors.actionDisabledText,
        ...(useNewShadows && {
            boxShadow: 'none',
        }),
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
            borderColor: 'transparent',
            color: theme.colors.actionDisabledText,
        },
        '&:active': {
            backgroundColor: theme.colors.actionTertiaryBackgroundDefault,
            borderColor: 'transparent',
            color: theme.colors.actionDisabledText,
        },
    };
}
export function getDisabledSplitButtonStyles(theme, useNewShadows) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionDefaultBackgroundDefault,
        borderColor: theme.colors.actionDisabledBorder,
        color: theme.colors.actionDisabledText,
        ...(useNewShadows && {
            boxShadow: 'none',
        }),
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundDefault,
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
        },
        '&:active': {
            backgroundColor: theme.colors.actionDefaultBackgroundDefault,
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
        },
    };
}
export function getDisabledPrimarySplitButtonStyles(theme, useNewShadows) {
    const defaultStyles = {
        backgroundColor: theme.colors.actionDisabledBorder,
        color: theme.colors.actionPrimaryTextDefault,
        ...(useNewShadows && {
            boxShadow: 'none',
        }),
    };
    return {
        ...defaultStyles,
        '&:hover': {
            backgroundColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionPrimaryTextDefault,
        },
        '&:active': {
            backgroundColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionPrimaryTextDefault,
        },
    };
}
//# sourceMappingURL=styles.js.map
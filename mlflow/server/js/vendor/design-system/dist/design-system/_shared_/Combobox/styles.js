import { css } from '@emotion/react';
import { getDialogComboboxOptionLabelWidth } from '../../DialogCombobox/shared';
import { getDarkModePortalStyles, importantify } from '../../utils/css-utils';
export const getComboboxContentWrapperStyles = (theme, { maxHeight = '100vh', maxWidth = '100vw', minHeight = 0, minWidth = 0, width, useNewShadows, useNewBorderColors, }) => {
    return css({
        maxHeight,
        maxWidth,
        minHeight,
        minWidth,
        ...(width ? { width } : {}),
        background: theme.colors.backgroundPrimary,
        color: theme.colors.textPrimary,
        overflow: 'auto',
        // Making sure the content popover overlaps the remove button when opens to the right
        zIndex: theme.options.zIndexBase + 10,
        boxSizing: 'border-box',
        border: `1px solid ${theme.colors.border}`,
        boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
        borderRadius: theme.borders.borderRadiusSm,
        colorScheme: theme.isDarkMode ? 'dark' : 'light',
        ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
    });
};
export const COMBOBOX_MENU_ITEM_PADDING = [6, 32, 6, 12];
export const getComboboxOptionItemWrapperStyles = (theme) => {
    return css(importantify({
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'flex-start',
        justifyContent: 'flex-start',
        alignSelf: 'stretch',
        padding: `${COMBOBOX_MENU_ITEM_PADDING.map((x) => `${x}px`).join(' ')}`,
        lineHeight: theme.typography.lineHeightBase,
        boxSizing: 'content-box',
        cursor: 'pointer',
        userSelect: 'none',
        '&[data-highlighted="true"]': { background: theme.colors.actionTertiaryBackgroundHover },
        '&:focus': {
            background: theme.colors.actionTertiaryBackgroundHover,
            outline: 'none',
        },
        '&[disabled]': {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText,
            background: theme.colors.backgroundPrimary,
        },
    }));
};
export const getComboboxOptionLabelStyles = ({ theme, dangerouslyHideCheck, textOverflowMode, contentWidth, hasHintColumn, }) => {
    return css({
        marginLeft: !dangerouslyHideCheck ? theme.spacing.sm : 0,
        fontSize: theme.typography.fontSizeBase,
        fontStyle: 'normal',
        fontWeight: 400,
        cursor: 'pointer',
        overflow: 'hidden',
        wordBreak: 'break-word',
        ...(textOverflowMode === 'ellipsis' && {
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
        }),
        ...(contentWidth ? { width: getDialogComboboxOptionLabelWidth(theme, contentWidth) } : {}),
        ...(hasHintColumn && { display: 'flex' }),
    });
};
export const getInfoIconStyles = (theme) => {
    return css({
        paddingLeft: theme.spacing.xs,
        color: theme.colors.textSecondary,
        pointerEvents: 'all',
        cursor: 'pointer',
        verticalAlign: 'middle',
    });
};
export const getCheckboxStyles = (theme, textOverflowMode) => {
    return css({
        pointerEvents: 'none',
        height: 'unset',
        width: '100%',
        '& > label': {
            display: 'flex',
            width: '100%',
            fontSize: theme.typography.fontSizeBase,
            fontStyle: 'normal',
            fontWeight: 400,
            cursor: 'pointer',
            '& > span:first-of-type': {
                alignSelf: 'flex-start',
                display: 'inline-flex',
                alignItems: 'center',
                paddingTop: theme.spacing.xs / 2,
            },
            '& > span:last-of-type, & > span:last-of-type > label': {
                paddingRight: 0,
                width: '100%',
                overflow: 'hidden',
                wordBreak: 'break-word',
                ...(textOverflowMode === 'ellipsis'
                    ? {
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                    }
                    : {}),
            },
        },
    });
};
export const getFooterStyles = (theme) => {
    return css({
        width: '100%',
        background: theme.colors.backgroundPrimary,
        padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px ${theme.spacing.lg / 2}px`,
        position: 'sticky',
        bottom: 0,
        boxSizing: 'border-box',
        '&:has(> .combobox-footer-add-button)': {
            padding: `${theme.spacing.sm}px 0 ${theme.spacing.sm}px 0`,
            '& > :not(.combobox-footer-add-button)': {
                marginLeft: `${theme.spacing.lg / 2}px`,
                marginRight: `${theme.spacing.lg / 2}px`,
            },
            '& > .combobox-footer-add-button': {
                justifyContent: 'flex-start !important',
            },
        },
    });
};
export const getSelectItemWithHintColumnStyles = (hintColumnWidthPercent = 50) => {
    return css({
        flexGrow: 1,
        display: 'inline-grid',
        gridTemplateColumns: `${100 - hintColumnWidthPercent}% ${hintColumnWidthPercent}%`,
    });
};
export const getHintColumnStyles = (theme, disabled, textOverflowMode) => {
    return css({
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        textAlign: 'right',
        ...(disabled && {
            color: theme.colors.actionDisabledText,
        }),
        ...(textOverflowMode === 'ellipsis' && {
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
        }),
    });
};
//# sourceMappingURL=styles.js.map
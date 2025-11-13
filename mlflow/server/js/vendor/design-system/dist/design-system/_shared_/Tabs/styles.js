export const getCommonTabsListStyles = (theme) => {
    return {
        display: 'flex',
        borderBottom: `1px solid ${theme.colors.border}`,
        marginBottom: theme.spacing.md,
        height: theme.general.heightSm,
        boxSizing: 'border-box',
    };
};
export const getCommonTabsTriggerStyles = (theme) => {
    return {
        display: 'flex',
        fontWeight: theme.typography.typographyBoldFontWeight,
        fontSize: theme.typography.fontSizeMd,
        backgroundColor: 'transparent',
        marginRight: theme.spacing.md,
    };
};
//# sourceMappingURL=styles.js.map
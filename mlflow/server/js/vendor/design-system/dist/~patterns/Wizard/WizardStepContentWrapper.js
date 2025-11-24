import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { getShadowScrollStyles, Typography, useDesignSystemTheme } from '../../design-system';
export function WizardStepContentWrapper({ header, title, description, alertContent, children, }) {
    const { theme } = useDesignSystemTheme();
    return (_jsxs("div", { css: {
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
        }, children: [_jsxs("div", { style: {
                    backgroundColor: theme.colors.backgroundSecondary,
                    padding: theme.spacing.lg,
                    display: 'flex',
                    flexDirection: 'column',
                    borderTopLeftRadius: theme.legacyBorders.borderRadiusLg,
                    borderTopRightRadius: theme.legacyBorders.borderRadiusLg,
                }, children: [_jsx(Typography.Text, { size: "sm", style: { fontWeight: 500 }, children: header }), _jsx(Typography.Title, { withoutMargins: true, style: { paddingTop: theme.spacing.lg }, level: 3, children: title }), _jsx(Typography.Text, { color: "secondary", children: description })] }), alertContent && (_jsx("div", { css: {
                    padding: `${theme.spacing.lg}px ${theme.spacing.lg}px 0`,
                }, children: alertContent })), _jsx("div", { css: {
                    display: 'flex',
                    flexDirection: 'column',
                    height: '100%',
                    padding: `${theme.spacing.lg}px ${theme.spacing.lg}px 0`,
                    overflowY: 'auto',
                    ...getShadowScrollStyles(theme),
                }, children: children })] }));
}
//# sourceMappingURL=WizardStepContentWrapper.js.map
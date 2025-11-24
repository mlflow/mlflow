import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDesignSystemTheme } from '../../Hooks';
export const SectionHeader = ({ children, ...props }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { ...props, css: {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'flex-start',
            padding: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
            alignSelf: 'stretch',
            fontWeight: 400,
            color: theme.colors.textSecondary,
        }, children: children }));
};
//# sourceMappingURL=SectionHeader.js.map
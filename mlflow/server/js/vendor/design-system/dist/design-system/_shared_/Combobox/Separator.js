import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDesignSystemTheme } from '../../Hooks';
export const Separator = (props) => {
    const { theme } = useDesignSystemTheme();
    return (_jsx("div", { ...props, css: {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            margin: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
            border: `1px solid ${theme.colors.borderDecorative}`,
            borderBottom: 0,
            alignSelf: 'stretch',
        } }));
};
//# sourceMappingURL=Separator.js.map
import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDialogComboboxContext } from '../../DialogCombobox/hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../../Hooks';
export const EmptyResults = ({ emptyText }) => {
    const { theme } = useDesignSystemTheme();
    const { emptyText: emptyTextFromContext } = useDialogComboboxContext();
    return (_jsx("div", { "aria-live": "assertive", css: {
            color: theme.colors.textSecondary,
            textAlign: 'center',
            padding: '6px 12px',
            width: '100%',
            boxSizing: 'border-box',
        }, children: emptyTextFromContext ?? emptyText ?? 'No results found' }));
};
//# sourceMappingURL=EmptyResults.js.map
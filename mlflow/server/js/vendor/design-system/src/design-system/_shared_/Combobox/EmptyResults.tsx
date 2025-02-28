import { useDialogComboboxContext } from '../../DialogCombobox/hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../../Hooks';

export interface EmptyResultsProps {
  emptyText?: string | React.ReactNode;
}

export const EmptyResults = ({ emptyText }: EmptyResultsProps) => {
  const { theme } = useDesignSystemTheme();
  const { emptyText: emptyTextFromContext } = useDialogComboboxContext();

  return (
    <div
      aria-live="assertive"
      css={{
        color: theme.colors.textSecondary,
        textAlign: 'center',
        padding: '6px 12px',
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      {emptyTextFromContext ?? emptyText ?? 'No results found'}
    </div>
  );
};

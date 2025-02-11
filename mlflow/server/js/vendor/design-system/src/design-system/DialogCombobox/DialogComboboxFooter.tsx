import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../Hooks';
import { getFooterStyles } from '../_shared_/Combobox';

export const DialogComboboxFooter = ({ children, ...restProps }: React.HTMLAttributes<HTMLDivElement>) => {
  const { theme } = useDesignSystemTheme();
  const { isInsideDialogCombobox } = useDialogComboboxContext();

  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxFooter` must be used within `DialogCombobox`');
  }

  return (
    <div {...restProps} css={getFooterStyles(theme)}>
      {children}
    </div>
  );
};

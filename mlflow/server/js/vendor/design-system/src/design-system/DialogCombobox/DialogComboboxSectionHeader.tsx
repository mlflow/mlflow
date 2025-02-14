import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { SectionHeader } from '../_shared_/Combobox';

export const DialogComboboxSectionHeader = ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  const { isInsideDialogCombobox } = useDialogComboboxContext();

  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxSectionHeader` must be used within `DialogCombobox`');
  }

  return <SectionHeader {...props}>{children}</SectionHeader>;
};

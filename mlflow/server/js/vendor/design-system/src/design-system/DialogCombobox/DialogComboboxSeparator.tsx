import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { Separator } from '../_shared_/Combobox';

export const DialogComboboxSeparator = (props: React.HTMLAttributes<HTMLDivElement>) => {
  const { isInsideDialogCombobox } = useDialogComboboxContext();

  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxSeparator` must be used within `DialogCombobox`');
  }

  return <Separator {...props} />;
};

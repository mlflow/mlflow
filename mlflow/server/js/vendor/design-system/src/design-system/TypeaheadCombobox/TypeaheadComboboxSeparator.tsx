import { useTypeaheadComboboxContext } from './hooks';
import { Separator } from '../_shared_/Combobox';

export const TypeaheadComboboxSeparator = (props: React.HTMLAttributes<HTMLDivElement>) => {
  const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();

  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxSeparator` must be used within `TypeaheadComboboxMenu`');
  }

  return <Separator {...props} />;
};

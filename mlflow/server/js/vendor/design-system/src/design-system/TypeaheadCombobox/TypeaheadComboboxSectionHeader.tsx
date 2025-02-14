import { useTypeaheadComboboxContext } from './hooks';
import { SectionHeader } from '../_shared_/Combobox';

export const TypeaheadComboboxSectionHeader = ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();

  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxSectionHeader` must be used within `TypeaheadComboboxMenu`');
  }

  return <SectionHeader {...props}>{children}</SectionHeader>;
};

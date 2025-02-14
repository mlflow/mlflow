import { DialogComboboxSectionHeader } from '../DialogCombobox';

export interface SelectOptionGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  name: string | React.ReactNode;
  children: React.ReactNode;
}

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectOptionGroup = (props: SelectOptionGroupProps) => {
  const { name, children, ...restProps } = props;
  return (
    <>
      <DialogComboboxSectionHeader {...restProps}>{name}</DialogComboboxSectionHeader>
      {children}
    </>
  );
};

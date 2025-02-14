import { SelectContextProvider } from './providers/SelectContext';
import type { ConditionalOptionalLabel, DialogComboboxProps } from '../DialogCombobox';
import { DialogCombobox } from '../DialogCombobox';

export interface SelectProps extends Omit<DialogComboboxProps, 'multiselect' | 'value'> {
  placeholder?: string;
  value?: string;
  id?: string;
}

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const Select = (props: SelectProps & ConditionalOptionalLabel) => {
  const { children, placeholder, value, label, ...restProps } = props;

  return (
    <SelectContextProvider value={{ isSelect: true, placeholder }}>
      <DialogCombobox label={label} value={value ? [value] : []} {...restProps}>
        {children}
      </DialogCombobox>
    </SelectContextProvider>
  );
};

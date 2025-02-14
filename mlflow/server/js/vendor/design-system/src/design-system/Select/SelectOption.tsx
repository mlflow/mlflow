import { forwardRef } from 'react';

import type { DialogComboboxOptionListSelectItemProps } from '../DialogCombobox';
import { DialogComboboxOptionListSelectItem } from '../DialogCombobox';
import { useDialogComboboxContext } from '../DialogCombobox/hooks/useDialogComboboxContext';

export interface SelectOptionProps extends DialogComboboxOptionListSelectItemProps {}

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectOption = forwardRef<HTMLDivElement, SelectOptionProps>((props, ref) => {
  const { value } = useDialogComboboxContext();
  return <DialogComboboxOptionListSelectItem checked={value && value[0] === props.value} {...props} ref={ref} />;
});

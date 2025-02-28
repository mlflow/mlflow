import { forwardRef } from 'react';

import type { DialogComboboxTriggerProps } from '../DialogCombobox';
import { DialogComboboxTrigger } from '../DialogCombobox';

export interface SelectTriggerProps extends DialogComboboxTriggerProps {}

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectTrigger = forwardRef<HTMLButtonElement, SelectTriggerProps>((props, ref) => {
  const { children, ...restProps } = props;
  return (
    <DialogComboboxTrigger allowClear={false} {...restProps} ref={ref}>
      {children}
    </DialogComboboxTrigger>
  );
});

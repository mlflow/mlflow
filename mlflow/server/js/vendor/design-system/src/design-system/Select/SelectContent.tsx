import { forwardRef } from 'react';

import type { DialogComboboxContentProps } from '../DialogCombobox';
import { DialogComboboxContent, DialogComboboxOptionList } from '../DialogCombobox';

export interface SelectContentProps extends DialogComboboxContentProps {}

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectContent = forwardRef<HTMLDivElement, SelectContentProps>(
  ({ children, minWidth = 150, ...restProps }, ref) => {
    return (
      <DialogComboboxContent minWidth={minWidth} {...restProps} ref={ref}>
        <DialogComboboxOptionList>{children}</DialogComboboxOptionList>
      </DialogComboboxContent>
    );
  },
);

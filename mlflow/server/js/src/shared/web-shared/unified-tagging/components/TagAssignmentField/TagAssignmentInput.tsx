import { forwardRef } from 'react';

import type { InputProps, InputRef } from '@databricks/design-system';
import { FormUI, Input } from '@databricks/design-system';

interface TagAssignmentInputProps extends InputProps {
  errorMessage?: string;
}

export const TagAssignmentInput: React.ForwardRefExoticComponent<
  TagAssignmentInputProps & React.RefAttributes<InputRef>
> = forwardRef<InputRef, TagAssignmentInputProps>(({ errorMessage, ...otherProps }: TagAssignmentInputProps, ref) => {
  return (
    <div css={{ flex: 1 }}>
      <Input validationState={errorMessage ? 'error' : 'info'} {...otherProps} ref={ref} />
      {errorMessage && <FormUI.Message message={errorMessage} type="error" />}
    </div>
  );
});

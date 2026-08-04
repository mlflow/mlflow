import { forwardRef } from 'react';
import { Input } from '@databricks/design-system';
import type { InputProps, InputRef } from '@databricks/design-system';

export const GatewayInput = forwardRef<InputRef, InputProps>(function GatewayInput(props, ref) {
  return (
    <Input
      ref={ref}
      {...props}
      autoComplete="off"
      data-1p-ignore="true"
      data-lpignore="true"
      data-keeper-ignore="true"
      data-form-type="other"
    />
  );
});

import { Input } from '@databricks/design-system';
import type { InputProps } from '@databricks/design-system';

export const GatewayInput = (props: InputProps) => {
  return (
    <Input
      {...props}
      autoComplete="off"
      data-1p-ignore="true"
      data-lpignore="true"
      data-keeper-ignore="true"
      data-form-type="other"
    />
  );
};

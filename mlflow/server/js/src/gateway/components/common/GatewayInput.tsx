import { Input } from '@databricks/design-system';
import type { InputProps } from '@databricks/design-system';

/**
 * A wrapper around the design system Input that disables password manager detection.
 * Use this for all text inputs in the gateway to prevent password managers like
 * 1Password, LastPass, and Keeper from offering to save/fill values.
 */
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

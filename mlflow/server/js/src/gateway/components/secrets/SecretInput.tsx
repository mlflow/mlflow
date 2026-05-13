import { useState, useCallback } from 'react';
import { Button, useDesignSystemTheme, VisibleIcon, VisibleOffIcon } from '@databricks/design-system';
import type { InputProps } from '@databricks/design-system';
import { GatewayInput } from '../common';

export interface SecretInputProps extends Omit<InputProps, 'type' | 'suffix'> {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export const SecretInput = ({ value, onChange, disabled, componentId, ...props }: SecretInputProps) => {
  const { theme } = useDesignSystemTheme();
  const [isVisible, setIsVisible] = useState(false);

  const toggleVisibility = useCallback(() => {
    setIsVisible((prev) => !prev);
  }, []);

  return (
    <GatewayInput
      {...props}
      componentId={componentId}
      type="text"
      css={{
        fontFamily: 'monospace',
        WebkitTextSecurity: isVisible ? 'none' : 'disc',
      }}
      value={value}
      onChange={onChange}
      disabled={disabled}
      suffix={
        <Button
          componentId={`${componentId}.toggle-visibility`}
          type="tertiary"
          size="small"
          icon={isVisible ? <VisibleOffIcon /> : <VisibleIcon />}
          onClick={toggleVisibility}
          disabled={disabled}
          aria-label={isVisible ? 'Hide value' : 'Show value'}
          data-1p-ignore="true"
          data-lpignore="true"
          data-keeper-ignore="true"
          data-form-type="other"
          css={{
            minWidth: 'auto',
            padding: theme.spacing.xs,
          }}
        />
      }
    />
  );
};

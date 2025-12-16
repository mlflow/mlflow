import { useState, useCallback } from 'react';
import { Button, useDesignSystemTheme, VisibleIcon, VisibleOffIcon } from '@databricks/design-system';
import type { InputProps } from '@databricks/design-system';
import { GatewayInput } from '../common';

interface SecretInputProps extends Omit<InputProps, 'type' | 'suffix'> {
  /** Current value */
  value: string;
  /** Callback when value changes */
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

/**
 * A specialized input component for entering sensitive values like API keys.
 *
 * Features:
 * - Toggle visibility with eye icon (hidden by default)
 * - Uses CSS text-security for masking instead of type="password"
 * - Prevents password manager detection with appropriate attributes
 * - Does NOT use type="password" to avoid browser password save prompts
 */
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
        // Use -webkit-text-security for masking (works in webkit browsers)
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
          css={{
            minWidth: 'auto',
            padding: theme.spacing.xs,
          }}
        />
      }
    />
  );
};

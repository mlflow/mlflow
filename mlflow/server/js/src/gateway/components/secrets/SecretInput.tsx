import { useState, useCallback } from 'react';
import { Button, FormUI, useDesignSystemTheme, VisibleIcon, VisibleOffIcon } from '@databricks/design-system';
import type { InputProps } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { GatewayInput } from '../common';
import { containsNonAsciiCharacters } from '../../utils/secretUtils';

export interface SecretInputProps extends Omit<InputProps, 'type' | 'suffix'> {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  /**
   * Show an inline error when the value contains non-ASCII characters. The input stays a text
   * field (masked via CSS), so the OS IME remains active and can silently compose keystrokes into
   * non-Latin text. Callers must also block submission; see `hasNonAsciiApiKey`.
   */
  asciiOnly?: boolean;
}

export const SecretInput = ({
  value,
  onChange,
  disabled,
  componentId,
  asciiOnly = false,
  validationState,
  ...props
}: SecretInputProps) => {
  const { theme } = useDesignSystemTheme();
  const [isVisible, setIsVisible] = useState(false);

  const toggleVisibility = useCallback(() => {
    setIsVisible((prev) => !prev);
  }, []);

  const hasNonAsciiError = asciiOnly && containsNonAsciiCharacters(value);

  return (
    <>
      <GatewayInput
        {...props}
        componentId={componentId}
        validationState={hasNonAsciiError ? 'error' : validationState}
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
      {hasNonAsciiError && <NonAsciiApiKeyMessage />}
    </>
  );
};

export const NonAsciiApiKeyMessage = () => {
  const intl = useIntl();
  return (
    <FormUI.Message
      type="error"
      message={intl.formatMessage({
        defaultMessage:
          'API key contains non-ASCII characters. Check that your keyboard is not in a non-Latin input mode.',
        description: 'Error shown when a masked API key input contains non-ASCII characters, e.g. from an IME',
      })}
    />
  );
};

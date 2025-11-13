import { Button, VisibleOffIcon, VisibleIcon, Input, useDesignSystemTheme } from '@databricks/design-system';
import { useState, useRef, useEffect } from 'react';

interface MaskedApiKeyInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  componentId?: string;
  id?: string;
}

/**
 * Custom masked input that:
 * - Does NOT use type="password" (avoids browser password managers)
 * - Shows only the last character typed, masks all previous characters
 * - Prevents autocomplete and password manager detection
 */
export const MaskedApiKeyInput = ({
  value,
  onChange,
  placeholder,
  componentId = 'masked-api-key-input',
  id,
}: MaskedApiKeyInputProps) => {
  const { theme } = useDesignSystemTheme();
  const [displayValue, setDisplayValue] = useState('');
  const [lastCharTimeout, setLastCharTimeout] = useState<NodeJS.Timeout | null>(null);
  const [showValue, setShowValue] = useState(false);
  const actualValueRef = useRef(value);

  // Update actual value ref when prop changes (from external source, not from typing)
  useEffect(() => {
    // Only update if the value actually changed (e.g., from parent reset, not from our own onChange)
    if (actualValueRef.current !== value) {
      actualValueRef.current = value;
      // Show all characters masked when value changes externally
      setDisplayValue('•'.repeat(value.length));
    }
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newDisplayValue = e.target.value;
    const prevLength = actualValueRef.current.length;
    const newLength = newDisplayValue.length;

    let newActualValue: string;

    if (newLength > prevLength) {
      // Character added - get the new character(s)
      const addedChars = newDisplayValue.substring(prevLength);
      newActualValue = actualValueRef.current + addedChars.replace(/•/g, '');

      // Show the last character for 500ms, then mask it
      setDisplayValue('•'.repeat(newActualValue.length - 1) + newActualValue.slice(-1));

      // Clear previous timeout
      if (lastCharTimeout) {
        clearTimeout(lastCharTimeout);
      }

      // Mask the last character after 500ms
      const timeout = setTimeout(() => {
        setDisplayValue('•'.repeat(newActualValue.length));
      }, 500);
      setLastCharTimeout(timeout);
    } else if (newLength < prevLength) {
      // Character(s) deleted
      const charsDeleted = prevLength - newLength;
      newActualValue = actualValueRef.current.substring(0, actualValueRef.current.length - charsDeleted);
      setDisplayValue('•'.repeat(newActualValue.length));
    } else {
      // Length same, probably pasted or replaced
      newActualValue = newDisplayValue.replace(/•/g, actualValueRef.current.slice(0, newDisplayValue.length));
      setDisplayValue('•'.repeat(newActualValue.length));
    }

    actualValueRef.current = newActualValue;
    onChange(newActualValue);
  };

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (lastCharTimeout) {
        clearTimeout(lastCharTimeout);
      }
    };
  }, [lastCharTimeout]);

  return (
    <div css={{ position: 'relative', width: '100%' }}>
      <Input
        componentId={componentId}
        id={id}
        type="text"
        // Prevent password managers from detecting this
        autoComplete="off"
        data-form-type="other"
        data-lpignore="true"
        data-1p-ignore="true"
        data-bwignore="true"
        name={`api-key-${Math.random()}`} // Random name to prevent autocomplete
        placeholder={placeholder}
        value={showValue ? actualValueRef.current : displayValue}
        onChange={handleChange}
        css={{
          fontFamily: 'monospace',
          letterSpacing: showValue ? 'normal' : '0.1em',
          paddingRight: theme.spacing.lg * 2,
        }}
      />
      <Button
        componentId={`${componentId}-toggle`}
        type="tertiary"
        size="small"
        icon={showValue ? <VisibleIcon /> : <VisibleOffIcon />}
        onClick={() => setShowValue(!showValue)}
        css={{
          position: 'absolute',
          right: theme.spacing.xs,
          top: '50%',
          transform: 'translateY(-50%)',
        }}
        aria-label={showValue ? 'Hide value' : 'Show value'}
      />
    </div>
  );
};

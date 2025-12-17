import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { formatCredentialFieldName } from '../../utils/providerUtils';

/**
 * Parses a masked value string that may be JSON or a simple value.
 * Returns an array of {key, value} pairs for multi-field secrets,
 * or null for simple single-value secrets.
 */
export function parseMaskedValue(maskedValue: string): Array<{ key: string; value: string }> | null {
  // Try to parse as JSON first
  try {
    const parsed = JSON.parse(maskedValue);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      const entries = Object.entries(parsed);
      // If there's only one entry and the value looks like a masked token (e.g., "abc...xyz"),
      // treat it as a simple value
      if (entries.length === 1) {
        const [, value] = entries[0];
        const strValue = String(value);
        // Check if it looks like a simple masked token (no internal structure)
        if (!strValue.includes(':') && strValue.includes('...')) {
          return null;
        }
      }
      return entries.map(([key, value]) => ({
        key,
        value: String(value),
      }));
    }
    // If it's a simple string, return null to display as-is
    if (typeof parsed === 'string') {
      return null;
    }
  } catch {
    // Not valid JSON, check for object-like format: {key: value, key2: value2}
    const match = maskedValue.match(/^\{(.+)\}$/s);
    if (match) {
      const pairs = match[1].split(/,\s*/);
      const result: Array<{ key: string; value: string }> = [];
      for (const pair of pairs) {
        const colonIndex = pair.indexOf(':');
        if (colonIndex > 0) {
          const key = pair.slice(0, colonIndex).trim();
          const value = pair.slice(colonIndex + 1).trim();
          result.push({ key, value });
        }
      }
      // Only return structured result if we have multiple entries
      // For single entries, check if it looks like a multi-field secret
      if (result.length > 1) {
        return result;
      }
      if (result.length === 1 && !result[0].value.includes('...')) {
        return result;
      }
    }
  }
  return null;
}

/**
 * Strips JSON formatting from a simple masked value.
 * E.g., '{"api_key": "abc...xyz"}' -> 'abc...xyz'
 */
export function stripMaskedValueFormatting(maskedValue: string): string {
  try {
    const parsed = JSON.parse(maskedValue);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      const values = Object.values(parsed);
      if (values.length === 1) {
        return String(values[0]);
      }
    }
    if (typeof parsed === 'string') {
      return parsed;
    }
  } catch {
    // Not JSON, strip brackets if present
    const match = maskedValue.match(/^\{["\s]*([^"{}]+)["\s]*\}$/);
    if (match) {
      return match[1].trim();
    }
  }
  return maskedValue;
}

interface MaskedValueDisplayProps {
  maskedValue: string;
  /** If true, displays in a compact inline format */
  compact?: boolean;
}

/**
 * Displays a masked value, formatting multi-field secrets with
 * key names in grey and values in code style.
 */
export const MaskedValueDisplay = ({ maskedValue, compact = false }: MaskedValueDisplayProps) => {
  const { theme } = useDesignSystemTheme();
  const parsed = useMemo(() => parseMaskedValue(maskedValue), [maskedValue]);
  const strippedValue = useMemo(() => stripMaskedValueFormatting(maskedValue), [maskedValue]);

  if (!parsed) {
    // Simple single value - display as code (strip JSON formatting)
    return (
      <Typography.Text
        css={{
          fontFamily: 'monospace',
          fontSize: theme.typography.fontSizeSm,
          backgroundColor: theme.colors.tagDefault,
          padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
          borderRadius: theme.general.borderRadiusBase,
          width: 'fit-content',
        }}
      >
        {strippedValue}
      </Typography.Text>
    );
  }

  // Multi-field secret - display each field on its own line
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: compact ? 2 : theme.spacing.xs / 2 }}>
      {parsed.map(({ key, value }) => (
        <div key={key} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            {formatCredentialFieldName(key)}:
          </Typography.Text>
          <Typography.Text
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              backgroundColor: theme.colors.tagDefault,
              padding: `${theme.spacing.xs / 4}px ${theme.spacing.xs}px`,
              borderRadius: theme.general.borderRadiusBase,
            }}
          >
            {value}
          </Typography.Text>
        </div>
      ))}
    </div>
  );
};

import type { IntlShape } from 'react-intl';

export const getMetadataFilterField = (key: string): string | undefined => {
  if (/[`";\\]|--|\/\*|\*\//.test(key) || [...key].some((character) => character < ' ')) return undefined;
  // The search parser needs backticks around keys that are not dotted identifiers.
  return `request_metadata.${/^[A-Za-z_]\w*(?:\.\w+)*$/.test(key) ? key : `\`${key}\``}`;
};

export const getMetadataFilterError = (keys: string[], intl: IntlShape): string | undefined =>
  keys.some((key) => key && !getMetadataFilterField(key))
    ? intl.formatMessage({
        defaultMessage:
          'Metadata keys cannot contain backticks, double quotes, backslashes, comments, semicolons, or control characters.',
        description: 'Trace filter validation error for an unsupported metadata key',
      })
    : undefined;

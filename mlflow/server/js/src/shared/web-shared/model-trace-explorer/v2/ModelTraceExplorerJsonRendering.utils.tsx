/* eslint-disable @databricks/no-hardcoded-colors */
import { truncate } from 'lodash';

const MAX_STRING_LENGTH = 1000;

export type JsonColors = {
  key: string;
  string: string;
  number: string;
  boolean: string;
  null: string;
  punctuation: string;
};

export function getJsonColors(isDarkMode: boolean, textPrimary: string, textSecondary: string): JsonColors {
  // colors that match Prism's JSON syntax highlighting, this is what users are used to
  // + combination of theme/databricks-duotone-dark.ts and theme/databricks-light.ts
  return {
    key: isDarkMode ? '#5DFAFC' : '#39adb5',
    string: isDarkMode ? '#ffffff' : textPrimary,
    number: isDarkMode ? '#3AACE2' : '#f5871f',
    boolean: isDarkMode ? '#ffffff' : textPrimary,
    null: isDarkMode ? '#ffffff' : textPrimary,
    punctuation: textSecondary,
  };
}

export function renderPrimitiveValue(val: unknown, colors: JsonColors): React.ReactNode {
  if (val === null) {
    return <span css={{ color: colors.null }}>null</span>;
  }
  if (val === undefined) {
    return <span css={{ color: colors.null }}>undefined</span>;
  }
  if (typeof val === 'string') {
    const truncated = truncate(val, { length: MAX_STRING_LENGTH });
    return <span css={{ color: colors.string }}>"{truncated}"</span>;
  }
  if (typeof val === 'number') {
    if (Number.isNaN(val)) {
      return <span css={{ color: colors.null }}>NaN</span>;
    }
    if (!Number.isFinite(val)) {
      return <span css={{ color: colors.null }}>{val > 0 ? 'Infinity' : '-Infinity'}</span>;
    }
    return <span css={{ color: colors.number }}>{val}</span>;
  }
  if (typeof val === 'boolean') {
    return <span css={{ color: colors.boolean }}>{String(val)}</span>;
  }
  return <span css={{ color: colors.null }}>{String(val)}</span>;
}

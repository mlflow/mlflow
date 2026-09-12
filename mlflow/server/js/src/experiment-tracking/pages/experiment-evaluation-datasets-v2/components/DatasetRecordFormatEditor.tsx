import { useCallback, useEffect, useRef, useState } from 'react';
import yaml from 'js-yaml';
import { LazyJsonRecordEditor } from './LazyJsonRecordEditor';
import type { JsonRecordEditorProps } from './JsonRecordEditor';

export type DatasetRecordFormat = 'json' | 'yaml';

const isRecordObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

export const serializeRecordAsYaml = (value: string): string => {
  if (value.trim() === '') return '';
  try {
    const parsed = JSON.parse(value);
    return isRecordObject(parsed)
      ? yaml.safeDump(parsed, { noRefs: true, schema: yaml.JSON_SCHEMA, sortKeys: false })
      : value;
  } catch {
    // Preserve invalid JSON verbatim when changing views so in-progress edits are never lost.
    return value;
  }
};

export const canonicalizeYamlRecord = (value: string): string | undefined => {
  if (value.trim() === '') return JSON.stringify({}, null, 2);
  try {
    const parsed = yaml.safeLoad(value, { schema: yaml.JSON_SCHEMA });
    return isRecordObject(parsed) ? JSON.stringify(parsed, null, 2) : undefined;
  } catch {
    return undefined;
  }
};

interface DatasetRecordFormatEditorProps extends Omit<JsonRecordEditorProps, 'language' | 'onChange' | 'value'> {
  format: DatasetRecordFormat;
  /** Canonical JSON text owned by the record save/create state. */
  value: string;
  onChange: (value: string) => void;
}

/**
 * Adapts the record editor's canonical JSON state to an editable YAML view. Valid YAML
 * objects are immediately normalized back to JSON for the existing save and dirty-state
 * machinery; invalid YAML remains visible verbatim and keeps Save disabled.
 */
export const DatasetRecordFormatEditor = ({
  format,
  value,
  onChange,
  ...editorProps
}: DatasetRecordFormatEditorProps) => {
  const [yamlText, setYamlText] = useState(() => serializeRecordAsYaml(value));
  const lastEmittedCanonical = useRef<string>();

  useEffect(() => {
    if (format !== 'yaml') return;
    if (value === lastEmittedCanonical.current) {
      lastEmittedCanonical.current = undefined;
      return;
    }
    setYamlText(serializeRecordAsYaml(value));
  }, [format, value]);

  const handleYamlChange = useCallback(
    (next: string) => {
      setYamlText(next);
      const canonical = canonicalizeYamlRecord(next) ?? next;
      lastEmittedCanonical.current = canonical;
      onChange(canonical);
    },
    [onChange],
  );

  if (format === 'json') {
    return <LazyJsonRecordEditor {...editorProps} language="json" value={value} onChange={onChange} />;
  }

  return <LazyJsonRecordEditor {...editorProps} language="yaml" value={yamlText} onChange={handleYamlChange} />;
};

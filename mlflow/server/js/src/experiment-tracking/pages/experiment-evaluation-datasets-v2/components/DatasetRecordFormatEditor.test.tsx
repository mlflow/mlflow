import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import { DatasetRecordFormatEditor, canonicalizeYamlRecord, serializeRecordAsYaml } from './DatasetRecordFormatEditor';

jest.mock('./LazyJsonRecordEditor', () => ({
  LazyJsonRecordEditor: ({
    value,
    onChange,
    language,
    ariaLabel,
  }: {
    value: string;
    onChange: (value: string) => void;
    language: string;
    ariaLabel: string;
  }) => (
    <textarea
      aria-label={ariaLabel}
      data-language={language}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

describe('DatasetRecordFormatEditor', () => {
  test('renders canonical JSON as editable YAML and emits canonical JSON', () => {
    const onChange = jest.fn();
    render(
      <DatasetRecordFormatEditor
        format="yaml"
        value={JSON.stringify({ question: 'Hello', score: 1 }, null, 2)}
        onChange={onChange}
        ariaLabel="Record inputs"
      />,
    );

    const editor = screen.getByRole('textbox', { name: 'Record inputs' });
    expect(editor).toHaveAttribute('data-language', 'yaml');
    expect(editor).toHaveValue('question: Hello\nscore: 1\n');

    fireEvent.change(editor, { target: { value: 'question: Updated\nscore: 2\n' } });
    expect(onChange).toHaveBeenLastCalledWith(JSON.stringify({ question: 'Updated', score: 2 }, null, 2));
    expect(editor).toHaveValue('question: Updated\nscore: 2\n');
  });

  test('keeps invalid YAML verbatim so the parent state remains invalid', () => {
    const onChange = jest.fn();
    render(<DatasetRecordFormatEditor format="yaml" value="{}" onChange={onChange} ariaLabel="Record inputs" />);

    const editor = screen.getByRole('textbox', { name: 'Record inputs' });
    fireEvent.change(editor, { target: { value: 'question: [unterminated' } });
    expect(onChange).toHaveBeenLastCalledWith('question: [unterminated');
    expect(editor).toHaveValue('question: [unterminated');
  });
});

describe('record format conversion', () => {
  test('requires a mapping at the top level in both views', () => {
    expect(canonicalizeYamlRecord('- one\n- two\n')).toBeUndefined();
    expect(canonicalizeYamlRecord('plain text')).toBeUndefined();
    expect(canonicalizeYamlRecord('')).toBe(JSON.stringify({}, null, 2));
  });

  test('preserves invalid JSON while changing to YAML view', () => {
    expect(serializeRecordAsYaml('{invalid')).toBe('{invalid');
  });
});

import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { JsonEditor } from './JsonEditor';

const renderEditor = (props: Partial<Parameters<typeof JsonEditor>[0]> = {}) => {
  const onChange = jest.fn<(next: string) => void>();
  render(
    <DesignSystemProvider>
      <JsonEditor ariaLabel="Editor" value="" onChange={onChange} {...props} />
    </DesignSystemProvider>,
  );
  return { onChange };
};

describe('JsonEditor', () => {
  it('exposes an editable textbox via its aria-label', () => {
    renderEditor({ value: '{"a": 1}' });
    expect(screen.getByRole('textbox', { name: 'Editor' })).toHaveValue('{"a": 1}');
  });

  it('forwards edits through onChange', () => {
    const { onChange } = renderEditor({ value: '' });
    fireEvent.change(screen.getByRole('textbox', { name: 'Editor' }), { target: { value: '{}' } });
    expect(onChange).toHaveBeenLastCalledWith('{}');
  });

  it('renders a line number for each line', () => {
    renderEditor({ value: 'a\nb\nc' });
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('syntax-highlights JSON tokens into spans', () => {
    renderEditor({ value: '{"a": 1}' });
    // The highlighted layer renders the object key as its own token span.
    expect(screen.getByText('"a"')).toBeInTheDocument();
  });

  it('renders characters that are not yet valid tokens (an in-progress quote)', () => {
    // The textarea text is transparent, so the highlighted layer must reproduce
    // every character — including a lone, unbalanced quote — or it appears to vanish.
    renderEditor({ value: '  "loc' });
    expect(screen.getByRole('textbox', { name: 'Editor' })).toHaveValue('  "loc');
    // The unbalanced quote is emitted as its own plain-text span in the highlight layer.
    expect(screen.getAllByText('"', { exact: true }).length).toBeGreaterThanOrEqual(1);
  });

  it('renders the placeholder when empty', () => {
    renderEditor({ value: '', placeholder: 'Paste JSON' });
    expect(screen.getByPlaceholderText('Paste JSON')).toBeInTheDocument();
  });

  it('marks the textbox invalid when flagged', () => {
    renderEditor({ value: '{', invalid: true });
    expect(screen.getByRole('textbox', { name: 'Editor' })).toHaveAttribute('aria-invalid', 'true');
  });

  it('inserts two spaces (and does not move focus) when Tab is pressed', () => {
    const { onChange } = renderEditor({ value: 'abc' });
    const textbox = screen.getByRole('textbox', { name: 'Editor' }) as HTMLTextAreaElement;
    textbox.focus();
    textbox.setSelectionRange(0, 0);
    const notCancelled = fireEvent.keyDown(textbox, { key: 'Tab' });
    expect(notCancelled).toBe(false); // preventDefault was called → focus stays
    expect(onChange).toHaveBeenLastCalledWith('  abc');
  });

  it('dedents up to two leading spaces on Shift+Tab', () => {
    const { onChange } = renderEditor({ value: '    abc' });
    const textbox = screen.getByRole('textbox', { name: 'Editor' }) as HTMLTextAreaElement;
    textbox.focus();
    textbox.setSelectionRange(6, 6);
    fireEvent.keyDown(textbox, { key: 'Tab', shiftKey: true });
    expect(onChange).toHaveBeenLastCalledWith('  abc');
  });
});

import { expect, jest, test } from '@jest/globals';
import { useEffect } from 'react';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { FilterOp } from '@databricks/web-shared/traces-table';
import { SqlFilterEditor } from './SqlFilterEditor';

const mockDispose = jest.fn();
const mockModel = { getValueInRange: () => 'tags.' };
const mockEditor = { getModel: () => mockModel };
const mockRegister = jest.fn((_language: string, _provider: any) => ({ dispose: mockDispose }));
jest.mock(
  'monaco-editor',
  () => ({
    languages: {
      registerCompletionItemProvider: (...args: [string, any]) => mockRegister(...args),
      CompletionItemKind: { Field: 5 },
    },
  }),
  { virtual: true },
);
jest.mock('@monaco-editor/react', () => ({
  __esModule: true,
  default: MockEditor,
  loader: { config: jest.fn() },
}));
function MockEditor({ onMount }: { onMount: (editor: unknown) => void }) {
  useEffect(() => onMount(mockEditor), [onMount]);
  return null;
}

test('suggests quoted loaded keys only for its own model and unregisters completions', () => {
  const { unmount } = render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <SqlFilterEditor
          value=""
          onChange={jest.fn()}
          fields={[
            {
              id: 'tag',
              label: 'Tag',
              valueInput: 'text',
              operators: [FilterOp.EQUALS],
              keyOptions: [
                { value: 'release name', label: 'Release' },
                { value: 'bad`key', label: 'Invalid' },
              ],
            },
          ]}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  const provider = mockRegister.mock.calls.at(-1)?.[1];
  const position = { lineNumber: 1, column: 6 };
  expect(provider.provideCompletionItems(mockModel, position).suggestions).toEqual([
    expect.objectContaining({
      label: 'tags.`release name`',
      insertText: 'tags.`release name`',
      range: { startLineNumber: 1, endLineNumber: 1, startColumn: 1, endColumn: 6 },
    }),
  ]);
  expect(provider.provideCompletionItems({}, position).suggestions).toEqual([]);
  expect(screen.getByText(/The server validates syntax, fields, and operators/)).toBeInTheDocument();
  unmount();
  expect(mockDispose).toHaveBeenCalled();
});

import { expect, jest, test } from '@jest/globals';
import { useEffect } from 'react';
import { render, renderHook, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { FilterOp } from '@databricks/web-shared/traces-table';
import { SqlFilterEditor } from './SqlFilterEditor';
import { parseBasicTraceFilter } from '../utils/traceSqlFilters';
import { useMlflowTraceFilterFields } from '../utils/filterModel';

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

test('session completion round trips to the Basic session field', () => {
  const { result } = renderHook(() => useMlflowTraceFilterFields(), {
    wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
  });
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <SqlFilterEditor value="" onChange={jest.fn()} fields={result.current} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  const provider = mockRegister.mock.calls.at(-1)?.[1];
  // Use the same model registered by the editor with an empty completion prefix.
  const original = mockModel.getValueInRange;
  mockModel.getValueInRange = () => '';
  const column = provider
    .provideCompletionItems(mockModel, { lineNumber: 1, column: 1 })
    .suggestions.find(({ label }: { label: string }) => label.includes('mlflow.trace.session')).insertText;
  mockModel.getValueInRange = original;
  expect(parseBasicTraceFilter(`${column} = 'session-1'`, result.current)).toEqual([
    { field: 'session', key: undefined, operator: FilterOp.EQUALS, value: 'session-1' },
  ]);
});

test.each(['(', '((', "tags.env = 'qa' AND ("])('replaces the column prefix after %s', (group) => {
  const before = `${group}attributes.`;
  const original = mockModel.getValueInRange;
  mockModel.getValueInRange = () => before;
  try {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <SqlFilterEditor value={before} onChange={jest.fn()} fields={[]} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    const provider = mockRegister.mock.calls.at(-1)?.[1];
    const completion = provider
      .provideCompletionItems(mockModel, { lineNumber: 1, column: before.length + 1 })
      .suggestions.find(({ label }: { label: string }) => label === 'attributes.status');
    expect(before.slice(0, completion.range.startColumn - 1) + completion.insertText).toBe(`${group}attributes.status`);
  } finally {
    mockModel.getValueInRange = original;
  }
});

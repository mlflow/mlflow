import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import type { ReactNode } from 'react';
import { DraftTagsField } from './DraftTagsField';

// The modal is exercised through its own tests; here we swap it for a thin harness that
// surfaces props and lets tests drive `onSave` / `onDelete` directly. Mirrors the harness
// used in TagsCell.test.tsx so the two suites stay structurally comparable.
jest.mock('../../experiment-evaluation-datasets/components/KeyValueTagFullViewModal', () => ({
  KeyValueTagFullViewModal: ({
    tagKey,
    tagValue,
    isKeyValueTagFullViewModalVisible,
    setIsKeyValueTagFullViewModalVisible,
    onSave,
    onDelete,
  }: {
    tagKey: string;
    tagValue: string;
    isKeyValueTagFullViewModalVisible: boolean;
    setIsKeyValueTagFullViewModalVisible: (v: boolean) => void;
    onSave?: (key: string, value: string) => Promise<void>;
    onDelete?: (key: string) => Promise<void>;
  }) =>
    isKeyValueTagFullViewModalVisible ? (
      <div role="dialog" aria-label="tag-modal">
        <span data-testid="modal-key">{tagKey}</span>
        <span data-testid="modal-value">{tagValue}</span>
        <button type="button" onClick={() => onSave?.(tagKey || 'new-key', tagValue || 'new-value')}>
          test-save
        </button>
        <button type="button" onClick={() => onSave?.('renamed-key', tagValue || 'new-value')}>
          test-save-rename
        </button>
        {onDelete ? (
          <button type="button" onClick={() => onDelete(tagKey)}>
            test-delete
          </button>
        ) : null}
        <button type="button" onClick={() => setIsKeyValueTagFullViewModalVisible(false)}>
          test-close
        </button>
      </div>
    ) : null,
}));

const renderField = (props: { tags: Record<string, string> }) => {
  const onChange = jest.fn<(next: Record<string, string>) => void>();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <IntlProvider locale="en">
      <DesignSystemProvider>{children}</DesignSystemProvider>
    </IntlProvider>
  );
  return {
    onChange,
    ...render(<DraftTagsField tags={props.tags} onChange={onChange} />, { wrapper }),
  };
};

describe('DraftTagsField', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders one pill per tag entry with key: value text', () => {
    renderField({ tags: { env: 'prod', region: 'us-west-2' } });
    expect(screen.getByText('env: prod')).toBeInTheDocument();
    expect(screen.getByText('region: us-west-2')).toBeInTheDocument();
  });

  test('empty state shows the labeled "Add tag" button', () => {
    renderField({ tags: {} });
    expect(screen.getByRole('button', { name: /^Add tag$/i })).toBeInTheDocument();
  });

  test('clicking the add button opens the modal in new-tag mode without delete', async () => {
    const user = userEvent.setup();
    renderField({ tags: {} });
    await user.click(screen.getByRole('button', { name: /^Add tag$/i }));
    expect(screen.getByRole('dialog', { name: 'tag-modal' })).toBeInTheDocument();
    expect(screen.getByTestId('modal-key')).toHaveTextContent('');
    expect(screen.getByTestId('modal-value')).toHaveTextContent('');
    // Add mode hides the delete affordance.
    expect(screen.queryByRole('button', { name: /^test-delete$/ })).not.toBeInTheDocument();
  });

  test('clicking an existing pill opens the modal in edit mode with prefill and delete', async () => {
    const user = userEvent.setup();
    renderField({ tags: { env: 'prod' } });
    await user.click(screen.getByText('env: prod'));
    expect(screen.getByTestId('modal-key')).toHaveTextContent('env');
    expect(screen.getByTestId('modal-value')).toHaveTextContent('prod');
    expect(screen.getByRole('button', { name: /^test-delete$/ })).toBeInTheDocument();
  });

  test('saving a new tag calls onChange with the merged tag map', async () => {
    const user = userEvent.setup();
    const { onChange } = renderField({ tags: { existing: 'value' } });
    await user.click(screen.getByRole('button', { name: /^Add tag$/i }));
    await user.click(screen.getByRole('button', { name: /^test-save$/ }));

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith({ existing: 'value', 'new-key': 'new-value' });
  });

  test('editing a tag with the same key updates the value in place', async () => {
    const user = userEvent.setup();
    const { onChange } = renderField({ tags: { env: 'prod' } });
    await user.click(screen.getByText('env: prod'));
    await user.click(screen.getByRole('button', { name: /^test-save$/ }));

    // Harness onSave uses `tagKey || 'new-key'` — for edit mode it submits the same key.
    expect(onChange).toHaveBeenCalledWith({ env: 'prod' });
  });

  test('renaming a tag drops the old key and writes the new one', async () => {
    const user = userEvent.setup();
    const { onChange } = renderField({ tags: { 'old-key': 'value', sibling: 'kept' } });
    await user.click(screen.getByText('old-key: value'));
    await user.click(screen.getByRole('button', { name: /^test-save-rename$/ }));

    expect(onChange).toHaveBeenCalledWith({ 'renamed-key': 'value', sibling: 'kept' });
  });

  test('clicking a pill close (X) calls onChange without the removed key', async () => {
    const user = userEvent.setup();
    const { onChange } = renderField({ tags: { a: '1', b: '2' } });
    const closeButtons = screen
      .getAllByRole('button')
      .filter(
        (btn) => btn.getAttribute('data-component-id') === 'mlflow.eval-datasets-v2.records.tag.draft.pill.close',
      );
    await user.click(closeButtons[0]);

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0][0] as Record<string, string>;
    expect(Object.keys(next)).toHaveLength(1);
    expect(next).not.toHaveProperty('a');
  });

  test('deleting via the modal calls onChange without the deleted key', async () => {
    const user = userEvent.setup();
    const { onChange } = renderField({ tags: { env: 'prod', region: 'us-west-2' } });
    await user.click(screen.getByText('env: prod'));
    await user.click(screen.getByRole('button', { name: /^test-delete$/ }));

    expect(onChange).toHaveBeenCalledWith({ region: 'us-west-2' });
  });
});

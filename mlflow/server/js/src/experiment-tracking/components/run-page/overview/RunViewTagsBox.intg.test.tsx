import userEvent from '@testing-library/user-event';

import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { renderWithIntl, fastFillInput, act, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { setRunTagsBulkApi } from '../../../actions';
import type { KeyValueEntity } from '../../../../common/types';
import { RunViewTagsBox } from './RunViewTagsBox';
import { DesignSystemProvider } from '@databricks/design-system';

const testRunUuid = 'test-run-uuid';

jest.mock('../../../actions', () => ({
  setRunTagsBulkApi: jest.fn(() => ({ type: 'setRunTagsBulkApi', payload: Promise.resolve() })),
}));

describe('RunViewTagsBox integration', () => {
  const onTagsUpdated = jest.fn();

  function renderTestComponent(existingTags: Record<string, KeyValueEntity> = {}) {
    renderWithIntl(
      <DesignSystemProvider>
        <MockedReduxStoreProvider>
          <RunViewTagsBox onTagsUpdated={onTagsUpdated} runUuid={testRunUuid} tags={existingTags} />,
        </MockedReduxStoreProvider>
      </DesignSystemProvider>,
    );
  }

  beforeEach(() => {
    jest.mocked(setRunTagsBulkApi).mockClear();
    onTagsUpdated.mockClear();
  });

  test('it should display empty tag list and adding a new one', async () => {
    // Render the component, wait to load initial data
    await act(async () => {
      renderTestComponent();
    });

    expect(screen.getByRole('button', { name: 'Add tags' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Add tags' }));

    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'new_tag_with_value');

    await userEvent.click(screen.getByText(/Add tag "new_tag_with_value"/));
    await fastFillInput(screen.getByLabelText('Value'), 'tag_value');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(setRunTagsBulkApi).toHaveBeenCalledWith(
      'test-run-uuid',
      [],
      [{ key: 'new_tag_with_value', value: 'tag_value' }],
    );
    expect(onTagsUpdated).toHaveBeenCalled();
  });

  test('should modify already existing tag list', async () => {
    // Render the component, wait to load initial data
    await act(async () => {
      renderTestComponent([
        { key: 'existing_tag_1', value: 'val1' },
        { key: 'existing_tag_2', value: 'val2' },
        { key: 'mlflow.existing_tag_3', value: 'val2' },
      ] as any);
    });

    expect(screen.getByRole('status', { name: 'existing_tag_1' })).toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'existing_tag_2' })).toBeInTheDocument();
    expect(screen.queryByRole('status', { name: /existing_tag_3/ })).not.toBeInTheDocument();

    expect(screen.getByRole('button', { name: 'Edit tags' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Edit tags' }));

    const modalBody = screen.getByRole('dialog');

    await userEvent.click(
      within(within(modalBody).getByRole('status', { name: 'existing_tag_1' })).getByRole('button'),
    );

    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'new_tag_with_value');

    await userEvent.click(screen.getByText(/Add tag "new_tag_with_value"/));
    await fastFillInput(screen.getByLabelText('Value'), 'tag_value');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(setRunTagsBulkApi).toHaveBeenCalledWith(
      'test-run-uuid',
      [
        { key: 'existing_tag_1', value: 'val1' },
        { key: 'existing_tag_2', value: 'val2' },
      ],
      [
        { key: 'existing_tag_2', value: 'val2' },
        { key: 'new_tag_with_value', value: 'tag_value' },
      ],
    );
    expect(onTagsUpdated).toHaveBeenCalled();
  });

  test('should react accordingly when API responds with an error', async () => {
    jest.mocked(setRunTagsBulkApi).mockImplementation(
      () =>
        ({
          type: 'setRunTagsBulkApi',
          payload: Promise.reject(new Error('Some error message')),
        } as any),
    );

    await act(async () => {
      renderTestComponent();
    });

    expect(screen.getByRole('button', { name: 'Add tags' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Add tags' }));

    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'new_tag_with_value');

    await userEvent.click(screen.getByText(/Add tag "new_tag_with_value"/));
    await fastFillInput(screen.getByLabelText('Value'), 'tag_value');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    expect(setRunTagsBulkApi).toHaveBeenCalledWith(
      'test-run-uuid',
      [],
      [{ key: 'new_tag_with_value', value: 'tag_value' }],
    );

    expect(screen.getByText('Some error message')).toBeInTheDocument();
  });
});

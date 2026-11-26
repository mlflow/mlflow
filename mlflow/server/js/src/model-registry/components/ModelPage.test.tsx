import { jest, describe, test, expect } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl, screen, waitFor } from '../../common/utils/TestUtils.react18';
import { ModelPageImpl } from './ModelPage';

jest.mock('../../common/components/RequestStateWrapper', () => ({
  ...jest.requireActual<typeof import('../../common/components/RequestStateWrapper')>(
    '../../common/components/RequestStateWrapper',
  ),
  __esModule: true,
  default: ({ children }: any) => children(false, false, []),
}));

jest.mock('./ModelView', () => ({
  ModelView: ({ onMetadataUpdated }: { onMetadataUpdated: () => void }) => (
    <button onClick={onMetadataUpdated}>Update metadata</button>
  ),
}));

describe('ModelPage', () => {
  test('onMetadataUpdated reloads the model without reusing the initial request IDs', async () => {
    const searchModelVersionsApi = jest.fn((..._args: any[]) => Promise.resolve({ value: {} }));
    const getRegisteredModelApi = jest.fn((..._args: any[]) => Promise.resolve({}));
    const props: any = {
      modelName: 'model-a',
      model: { name: 'model-a' },
      modelVersions: [],
      searchModelVersionsApi,
      getRegisteredModelApi,
      updateRegisteredModelApi: jest.fn(),
      deleteRegisteredModelApi: jest.fn(),
      navigate: jest.fn(),
      searchParams: new URLSearchParams(),
      setSearchParams: jest.fn(),
    };

    renderWithIntl(<ModelPageImpl {...props} />);

    await waitFor(() => expect(getRegisteredModelApi).toHaveBeenCalledTimes(1));
    expect(getRegisteredModelApi).toHaveBeenLastCalledWith('model-a', expect.any(String));

    await userEvent.click(screen.getByRole('button', { name: 'Update metadata' }));

    await waitFor(() => expect(getRegisteredModelApi).toHaveBeenCalledTimes(2));
    expect(getRegisteredModelApi).toHaveBeenLastCalledWith('model-a', null);
    expect(searchModelVersionsApi).toHaveBeenCalledTimes(2);
    expect(searchModelVersionsApi).toHaveBeenLastCalledWith(
      { name: 'model-a' },
      null,
      expect.any(Number),
      expect.any(String),
      null,
    );
  });
});

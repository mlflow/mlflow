import { describe, test, expect, jest } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import {
  IndexedDBInitializationContextProvider,
  IndexedDBInitializationContext,
} from './IndexedDBInitializationContext';
import { globalIndexedDBStorage } from '../../../common/utils/LocalStorageUtils';
import React, { useContext } from 'react';

jest.mock('../../../common/utils/LocalStorageUtils', () => ({
  globalIndexedDBStorage: {
    initialize: jest.fn(),
  },
}));

jest.mock('../../../common/utils/Utils', () => ({
  __esModule: true,
  default: {
    logErrorAndNotifyUser: jest.fn(),
  },
}));

describe('IndexedDBInitializationContext', () => {
  const TestComponent = () => {
    const { isIndexedDBAvailable } = useContext(IndexedDBInitializationContext);
    return <div>{isIndexedDBAvailable ? 'Available' : 'Unavailable'}</div>;
  };

  test('shows spinner while initializing', () => {
    (globalIndexedDBStorage.initialize as jest.Mock).mockImplementation(() => new Promise(() => {}));

    render(
      <IndexedDBInitializationContextProvider>
        <TestComponent />
      </IndexedDBInitializationContextProvider>,
    );

    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  test('sets isIndexedDBAvailable to true on successful initialization', async () => {
    // @ts-expect-error Argument of type 'undefined' is not assignable to parameter of type 'never'.
    (globalIndexedDBStorage.initialize as jest.Mock).mockResolvedValue(undefined);

    render(
      <IndexedDBInitializationContextProvider>
        <TestComponent />
      </IndexedDBInitializationContextProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Available')).toBeInTheDocument();
    });
  });

  test('sets isIndexedDBAvailable to false on initialization error', async () => {
    // @ts-expect-error Argument of type 'Error' is not assignable to parameter of type 'never'.
    (globalIndexedDBStorage.initialize as jest.Mock).mockRejectedValue(new Error('Init failed'));

    render(
      <IndexedDBInitializationContextProvider>
        <TestComponent />
      </IndexedDBInitializationContextProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Unavailable')).toBeInTheDocument();
    });
  });
});

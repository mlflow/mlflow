import { act, render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { TracesV4Tab } from './TracesV4Tab';

let mockProviderReady: boolean;
let mockProviderPromise: Promise<void>;
let mockResolveProvider: () => void;
const mockPageMount = jest.fn();

jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  shouldEnableModelTraceExplorerCustomTraceView: () => true,
}));

jest.mock('../traces-v3/ExperimentCustomViewProvider', () => ({
  ExperimentCustomViewProvider: ({ children }: { children: React.ReactNode }) => {
    if (!mockProviderReady) {
      throw mockProviderPromise;
    }
    return children;
  },
}));

jest.mock('./components/TracesV4PageContent', () => ({
  TracesV4PageContent: () => {
    const React = jest.requireActual<typeof import('react')>('react');
    React.useEffect(() => {
      mockPageMount();
    }, []);
    return <button>Interactive traces page</button>;
  },
}));

describe('TracesV4Tab', () => {
  beforeEach(() => {
    mockProviderReady = false;
    mockProviderPromise = new Promise<void>((resolve) => {
      mockResolveProvider = resolve;
    });
    mockPageMount.mockClear();
  });

  test('does not mount the interactive page while the Custom View provider is loading', async () => {
    render(
      <DesignSystemProvider>
        <TracesV4Tab experimentId="exp-1" />
      </DesignSystemProvider>,
    );

    expect(screen.queryByRole('button', { name: 'Interactive traces page' })).not.toBeInTheDocument();
    expect(mockPageMount).not.toHaveBeenCalled();

    await act(async () => {
      mockProviderReady = true;
      mockResolveProvider();
      await mockProviderPromise;
    });

    expect(await screen.findByRole('button', { name: 'Interactive traces page' })).toBeInTheDocument();
    expect(mockPageMount).toHaveBeenCalledTimes(1);
  });
});

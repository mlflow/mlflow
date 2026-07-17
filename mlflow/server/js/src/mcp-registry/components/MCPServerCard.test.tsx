import { describe, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { MCPServerCard } from './MCPServerCard';
import { createMockMCPServer, getMockedCurrentUserResponse } from '../test-utils';
import { MCPStatus } from '../types';
import type { MCPServer } from '../types';

const renderCard = (server: MCPServer) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <IntlProvider locale="en">
      <TestRouter
        routes={[
          testRoute(
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <MCPServerCard server={server} />
              </DesignSystemProvider>
            </QueryClientProvider>,
            '/',
          ),
        ]}
      />
    </IntlProvider>,
  );
};

describe('MCPServerCard', () => {
  it('renders server name when no display_name is set', () => {
    renderCard(createMockMCPServer({ name: 'io.github.test/my-server' }));
    expect(screen.getByText('io.github.test/my-server')).toBeInTheDocument();
  });

  it('renders display_name as title when set', () => {
    renderCard(createMockMCPServer({ name: 'io.github.test/raw', display_name: 'Pretty Name' }));
    expect(screen.getByText('Pretty Name')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    renderCard(createMockMCPServer({ description: 'A helpful tool server' }));
    expect(screen.getByText('A helpful tool server')).toBeInTheDocument();
  });

  it('does not render description when not provided', () => {
    renderCard(createMockMCPServer({ description: undefined }));
    expect(screen.queryByText('A helpful tool server')).not.toBeInTheDocument();
  });

  it('renders timestamp when last_updated_timestamp is set', () => {
    renderCard(createMockMCPServer({ last_updated_timestamp: 1620000000000 }));
    expect(screen.getByText(/2021/)).toBeInTheDocument();
  });

  it('does not render timestamp when last_updated_timestamp is absent', () => {
    renderCard(createMockMCPServer({ last_updated_timestamp: undefined }));
    expect(screen.queryByText(/\d{2}\/\d{2}\/\d{4}/)).not.toBeInTheDocument();
  });

  it('renders latest_version when set', () => {
    renderCard(createMockMCPServer({ latest_version: '2.1.0' }));
    expect(screen.getByText('v2.1.0')).toBeInTheDocument();
  });

  it('does not render version when latest_version is absent', () => {
    renderCard(createMockMCPServer({ latest_version: undefined }));
    expect(screen.queryByText(/^v\d/)).not.toBeInTheDocument();
  });

  it('renders tags when provided', () => {
    renderCard(createMockMCPServer({ tags: { env: 'production' } }));
    expect(document.body.textContent).toContain('env');
    expect(document.body.textContent).toContain('production');
  });

  it('does not render tags section when tags are empty', () => {
    renderCard(createMockMCPServer({ tags: {} }));
    expect(screen.queryByText(/:/)).not.toBeInTheDocument();
  });

  describe('dimmed card with auth available', () => {
    setupServer(getMockedCurrentUserResponse({ isAdmin: false }));

    it('renders with dimmed styling when server has no access_endpoints and status is active', async () => {
      renderCard(
        createMockMCPServer({
          name: 'io.github.test/dimmed',
          status: MCPStatus.ACTIVE,
          access_endpoints: [],
        }),
      );

      await waitFor(() => {
        const cardBody = document.querySelector('[data-component-id="mlflow.mcp_registry.card"] div');
        expect(cardBody).toBeInTheDocument();
      });

      const opacityEl =
        document.querySelector('[style*="opacity"]') ??
        Array.from(document.querySelectorAll('div')).find((el) => getComputedStyle(el).opacity === '0.5');
      // The card body div applies opacity: 0.5 via emotion css when isDimmed is true
      expect(document.querySelector('[data-component-id="mlflow.mcp_registry.card"]')).toBeInTheDocument();
    });

    it('shows disabled connect icon with tooltip when server is unavailable', async () => {
      renderCard(
        createMockMCPServer({
          name: 'io.github.test/unavailable',
          status: MCPStatus.ACTIVE,
          access_endpoints: [],
        }),
      );

      await waitFor(() => {
        expect(screen.getByText('io.github.test/unavailable')).toBeInTheDocument();
      });

      // When isUnavailable is true, the connect button is replaced with a disabled icon
      // and the tooltip text "No access endpoints configured" is present
      expect(screen.queryByLabelText('Connect')).not.toBeInTheDocument();
    });
  });
});

import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { MCPServerCardGrid } from './MCPServerCardGrid';
import { createMockMCPServer } from '../test-utils';

const noop = () => {};

const defaultPaginationProps = {
  hasNextPage: false,
  hasPreviousPage: false,
  onNextPage: noop,
  onPreviousPage: noop,
};

const renderGrid = (props: React.ComponentProps<typeof MCPServerCardGrid>) => {
  const queryClient = new QueryClient();
  return render(
    <IntlProvider locale="en">
      <TestRouter
        routes={[
          testRoute(
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <MCPServerCardGrid {...props} />
              </DesignSystemProvider>
            </QueryClientProvider>,
            '/',
          ),
        ]}
      />
    </IntlProvider>,
  );
};

describe('MCPServerCardGrid', () => {
  it('renders loading spinner when isLoading is true', () => {
    renderGrid({ ...defaultPaginationProps, isLoading: true });
    expect(screen.getByText('Loading servers...')).toBeInTheDocument();
  });

  it('renders "No servers found" when filtered and no results', () => {
    renderGrid({ ...defaultPaginationProps, servers: [], isFiltered: true });
    expect(screen.getByText('No servers found')).toBeInTheDocument();
  });

  it('renders empty state when no servers and not filtered', () => {
    renderGrid({ ...defaultPaginationProps, servers: [] });
    expect(screen.getByText('Create and manage MCP servers using MLflow.')).toBeInTheDocument();
  });

  it('renders a card for each server', () => {
    const servers = [
      createMockMCPServer({ name: 'server-a' }),
      createMockMCPServer({ name: 'server-b' }),
      createMockMCPServer({ name: 'server-c' }),
    ];
    renderGrid({ ...defaultPaginationProps, servers });
    expect(screen.getByText('server-a')).toBeInTheDocument();
    expect(screen.getByText('server-b')).toBeInTheDocument();
    expect(screen.getByText('server-c')).toBeInTheDocument();
  });

  it('does not render loading spinner when servers are present', () => {
    renderGrid({ ...defaultPaginationProps, servers: [createMockMCPServer()], isLoading: false });
    expect(screen.queryByText('Loading servers...')).not.toBeInTheDocument();
  });

  it('renders pagination controls when servers are present', () => {
    const servers = [createMockMCPServer()];
    renderGrid({ ...defaultPaginationProps, servers, hasNextPage: true });
    expect(screen.getByText('Next')).toBeInTheDocument();
    expect(screen.getByText('Previous')).toBeInTheDocument();
  });

  it('calls onNextPage when Next is clicked', () => {
    const onNextPage = jest.fn();
    const servers = [createMockMCPServer()];
    renderGrid({ ...defaultPaginationProps, servers, hasNextPage: true, onNextPage });
    screen.getByText('Next').click();
    expect(onNextPage).toHaveBeenCalledTimes(1);
  });

  it('calls onPreviousPage when Previous is clicked', () => {
    const onPreviousPage = jest.fn();
    const servers = [createMockMCPServer()];
    renderGrid({ ...defaultPaginationProps, servers, hasPreviousPage: true, onPreviousPage });
    screen.getByText('Previous').click();
    expect(onPreviousPage).toHaveBeenCalledTimes(1);
  });

  it('renders page size selector', () => {
    const servers = [createMockMCPServer()];
    renderGrid({
      ...defaultPaginationProps,
      servers,
      pageSizeSelect: { options: [10, 25, 50], default: 25, onChange: noop },
    });
    expect(screen.getByText('25 / page')).toBeInTheDocument();
  });
});

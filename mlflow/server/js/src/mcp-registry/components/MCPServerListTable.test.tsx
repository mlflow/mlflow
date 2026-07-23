import { describe, it, expect, jest } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { MCPServerListTable } from './MCPServerListTable';
import { createMockMCPServer } from '../test-utils';

const noop = () => {};

const renderTable = (props: Partial<React.ComponentProps<typeof MCPServerListTable>> = {}) => {
  const queryClient = new QueryClient();
  return render(
    <IntlProvider locale="en">
      <TestRouter
        routes={[
          testRoute(
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <MCPServerListTable
                  hasNextPage={false}
                  hasPreviousPage={false}
                  onNextPage={noop}
                  onPreviousPage={noop}
                  {...props}
                />
              </DesignSystemProvider>
            </QueryClientProvider>,
            '/',
          ),
        ]}
      />
    </IntlProvider>,
  );
};

describe('MCPServerListTable', () => {
  it('renders all column headers', () => {
    renderTable();
    expect(screen.getByText('Name')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByText('Latest version')).toBeInTheDocument();
    expect(screen.getByText('Last modified')).toBeInTheDocument();
    expect(screen.getByText('Tags')).toBeInTheDocument();
  });

  it('renders server rows with name and description', () => {
    const servers = [
      createMockMCPServer({
        name: 'io.github.test/server-a',
        description: 'A test server',
        last_updated_timestamp: 1620000000000,
      }),
      createMockMCPServer({
        name: 'io.github.test/server-b',
        description: 'Another test server',
      }),
    ];
    renderTable({ servers });
    expect(screen.getByText('io.github.test/server-a')).toBeInTheDocument();
    expect(screen.getByText('A test server')).toBeInTheDocument();
    expect(screen.getByText('io.github.test/server-b')).toBeInTheDocument();
    expect(screen.getByText('Another test server')).toBeInTheDocument();
  });

  it('falls back to name when display_name is absent', () => {
    const servers = [createMockMCPServer({ name: 'io.github.test/raw-name', display_name: undefined })];
    renderTable({ servers });
    expect(screen.getByText('io.github.test/raw-name')).toBeInTheDocument();
  });

  it('does not render data rows when loading', () => {
    renderTable({ isLoading: true, servers: [] });
    expect(screen.queryByText('Server A')).not.toBeInTheDocument();
    const allRows = screen.getAllByRole('row');
    expect(allRows.length).toBeGreaterThan(1);
  });

  it('renders empty state when no servers and not filtered', () => {
    renderTable({ servers: [] });
    expect(screen.getByText('Create and manage MCP servers using MLflow.')).toBeInTheDocument();
  });

  it('renders no-results state when filtered and empty', () => {
    renderTable({ servers: [], isFiltered: true });
    expect(screen.getByText('No servers found')).toBeInTheDocument();
  });

  it('renders server name in the table row', () => {
    const servers = [createMockMCPServer({ name: 'io.github.test/my-server' })];
    renderTable({ servers });
    expect(screen.getByText('io.github.test/my-server')).toBeInTheDocument();
  });

  it('renders pagination controls', () => {
    const servers = [createMockMCPServer()];
    renderTable({ servers, hasNextPage: true });
    expect(screen.getByText('Next')).toBeInTheDocument();
    expect(screen.getByText('Previous')).toBeInTheDocument();
  });

  it('calls onNextPage when Next is clicked', () => {
    const onNextPage = jest.fn();
    const servers = [createMockMCPServer()];
    renderTable({ servers, hasNextPage: true, onNextPage });
    screen.getByText('Next').click();
    expect(onNextPage).toHaveBeenCalledTimes(1);
  });

  it('calls onPreviousPage when Previous is clicked', () => {
    const onPreviousPage = jest.fn();
    const servers = [createMockMCPServer()];
    renderTable({ servers, hasPreviousPage: true, onPreviousPage });
    screen.getByText('Previous').click();
    expect(onPreviousPage).toHaveBeenCalledTimes(1);
  });

  it('renders latest version column', () => {
    const servers = [createMockMCPServer({ latest_version: '3.2.1' })];
    renderTable({ servers });
    expect(screen.getByText('3.2.1')).toBeInTheDocument();
  });

  it('renders em-dash when latest version is absent', () => {
    const servers = [createMockMCPServer({ latest_version: undefined })];
    renderTable({ servers });
    const cells = screen.getAllByText('—');
    expect(cells.length).toBeGreaterThanOrEqual(1);
  });

  it('renders tags column', () => {
    const servers = [createMockMCPServer({ tags: { env: 'staging' } })];
    renderTable({ servers });
    expect(document.body.textContent).toContain('env');
    expect(document.body.textContent).toContain('staging');
  });

  it('renders last modified column with formatted timestamp', () => {
    const servers = [createMockMCPServer({ last_updated_timestamp: 1620000000000 })];
    renderTable({ servers });
    const rows = screen.getAllByRole('row');
    const dataRow = rows[rows.length - 1];
    expect(dataRow.textContent).toMatch(/2021/);
  });

  it('does not show tooltip on description hover when not truncated', () => {
    const servers = [createMockMCPServer({ description: 'Short' })];
    renderTable({ servers });
    const descriptionCell = screen.getByText('Short');

    // scrollWidth equals clientWidth means no truncation
    Object.defineProperty(descriptionCell, 'scrollWidth', { value: 100, configurable: true });
    Object.defineProperty(descriptionCell, 'clientWidth', { value: 100, configurable: true });

    fireEvent.mouseEnter(descriptionCell);

    // Only one instance of the text (no tooltip duplicate)
    expect(screen.getAllByText('Short').length).toBe(1);
  });
});

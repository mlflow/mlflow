import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MCPServerTagsBox } from './MCPServerTagsBox';
import { createMockMCPServer } from '../test-utils';

const renderTagsBox = (props: Partial<React.ComponentProps<typeof MCPServerTagsBox>> = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MCPServerTagsBox {...props} />
        </DesignSystemProvider>
      </IntlProvider>
    </QueryClientProvider>,
  );
};

describe('MCPServerTagsBox', () => {
  it('renders "Add tags" button when server has no tags', () => {
    const server = createMockMCPServer({ tags: {} });
    renderTagsBox({ server });
    expect(screen.getByText('Add tags')).toBeInTheDocument();
  });

  it('renders tag values when server has tags', () => {
    const server = createMockMCPServer({ tags: { env: 'prod', team: 'ml' } });
    renderTagsBox({ server });
    expect(screen.getByText('env')).toBeInTheDocument();
    expect(screen.getByText('team')).toBeInTheDocument();
  });

  it('renders edit icon button when tags exist', () => {
    const server = createMockMCPServer({ tags: { env: 'prod' } });
    renderTagsBox({ server });
    expect(screen.getByRole('button', { name: 'Edit tags' })).toBeInTheDocument();
  });

  it('renders Add tags button when server is undefined', () => {
    renderTagsBox({ server: undefined });
    expect(screen.getByText('Add tags')).toBeInTheDocument();
  });
});

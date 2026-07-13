import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { MCPServerVersionList } from './MCPServerVersionList';
import { createMockMCPServerVersion } from '../test-utils';

const renderVersionList = (props: Partial<React.ComponentProps<typeof MCPServerVersionList>> = {}) => {
  const defaultProps = {
    versions: [],
    onSelectVersion: jest.fn(),
    serverDisplayName: 'Test Server',
    aliasesByVersion: {},
    ...props,
  };
  return render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MCPServerVersionList {...defaultProps} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

describe('MCPServerVersionList', () => {
  it('renders empty state when no versions exist', () => {
    renderVersionList({ versions: [] });
    expect(screen.getByText('No versions')).toBeInTheDocument();
  });

  it('renders version numbers', () => {
    const versions = [
      createMockMCPServerVersion({ version: '2.0.0' }),
      createMockMCPServerVersion({ version: '1.0.0' }),
    ];
    renderVersionList({ versions });
    expect(screen.getByText('2.0.0')).toBeInTheDocument();
    expect(screen.getByText('1.0.0')).toBeInTheDocument();
  });

  it('renders status tags', () => {
    const versions = [
      createMockMCPServerVersion({ version: '1.0.0', status: 'active' }),
      createMockMCPServerVersion({ version: '0.9.0', status: 'draft' }),
    ];
    renderVersionList({ versions });
    expect(screen.getByText('active')).toBeInTheDocument();
    expect(screen.getByText('draft')).toBeInTheDocument();
  });

  it('calls onSelectVersion when a row is clicked', async () => {
    const onSelectVersion = jest.fn();
    const versions = [createMockMCPServerVersion({ version: '1.0.0' })];
    renderVersionList({ versions, onSelectVersion });
    await userEvent.click(screen.getByText('1.0.0'));
    expect(onSelectVersion).toHaveBeenCalledWith('1.0.0');
  });

  it('highlights the selected version row', () => {
    const versions = [
      createMockMCPServerVersion({ version: '2.0.0' }),
      createMockMCPServerVersion({ version: '1.0.0' }),
    ];
    renderVersionList({ versions, selectedVersion: '1.0.0' });
    const selectedRow = screen.getByRole('row', { selected: true });
    expect(selectedRow).toBeInTheDocument();
  });

  it('renders version display name when different from server name', () => {
    const versions = [
      createMockMCPServerVersion({
        version: '1.0.0',
        display_name: 'Custom Name',
      }),
    ];
    renderVersionList({ versions, serverDisplayName: 'Server Name' });
    expect(screen.getByText('Custom Name')).toBeInTheDocument();
  });
});

import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { MCPServerVersionCompare } from './MCPServerVersionCompare';
import { createMockMCPServerVersion } from '../test-utils';
import { MCPStatus, TransportType } from '../types';

const v1 = createMockMCPServerVersion({
  version: '1',
  status: MCPStatus.ACTIVE,
  display_name: 'Alpha Server',
  source: 'https://github.com/org/alpha',
  server_json: {
    name: 'test',
    version: '1.0',
    title: 'Version 1',
    description: 'First version',
    remotes: [{ type: TransportType.SSE, url: 'https://example.com/v1' }],
  },
  tools: [{ name: 'search', description: 'Search the web' }],
  tags: { env: 'prod' },
  created_by: 'alice',
});
const v2 = createMockMCPServerVersion({
  version: '2',
  status: MCPStatus.DRAFT,
  display_name: 'Beta Server',
  source: 'https://github.com/org/beta',
  server_json: {
    name: 'test',
    version: '2.0',
    title: 'Version 2',
    description: 'Second version',
    remotes: [{ type: TransportType.STREAMABLE_HTTP, url: 'https://example.com/v2' }],
    packages: [{ registryName: 'npm', name: '@test/server' }] as any,
  },
  tools: [
    { name: 'search', description: 'Search the web v2' },
    { name: 'fetch', description: 'Fetch a URL' },
  ],
  tags: {},
  created_by: 'bob',
});

const renderCompare = (props: Partial<React.ComponentProps<typeof MCPServerVersionCompare>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MCPServerVersionCompare
          baselineVersion={v1}
          comparedVersion={v2}
          aliasesByVersion={{}}
          onSwitchSides={jest.fn()}
          {...props}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('MCPServerVersionCompare', () => {
  it('renders comparing heading with version numbers', () => {
    renderCompare();
    expect(screen.getByText(/Comparing version 1 with version 2/)).toBeInTheDocument();
  });

  it('renders status tags for both versions in metadata grid', () => {
    renderCompare();
    expect(screen.getByText('active')).toBeInTheDocument();
    expect(screen.getByText('draft')).toBeInTheDocument();
  });

  it('renders diff panels for changed fields', () => {
    renderCompare();
    expect(screen.getByText('Display name')).toBeInTheDocument();
    expect(screen.getByText('Source')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByText('Official endpoints')).toBeInTheDocument();
    expect(screen.getByText('Local packages')).toBeInTheDocument();
  });

  it('renders tools diff panel with content', () => {
    renderCompare();
    expect(screen.getByText('Tools')).toBeInTheDocument();
    expect(screen.getByText(/"name": "fetch"/)).toBeInTheDocument();
  });

  it('shows tools section even when both versions have no tools', () => {
    const noToolsV1 = createMockMCPServerVersion({ version: '1', tools: [] });
    const noToolsV2 = createMockMCPServerVersion({ version: '2', tools: undefined });
    renderCompare({ baselineVersion: noToolsV1, comparedVersion: noToolsV2 });
    expect(screen.getByText('Tools')).toBeInTheDocument();
  });

  it('collapses identical fields into a label list', () => {
    const same = createMockMCPServerVersion({
      version: '1',
      display_name: 'Same Name',
      source: 'https://same.example.com',
      server_json: { name: 'test', version: '1.0', title: 'Same', description: 'Same desc' },
    });
    const sameV2 = createMockMCPServerVersion({
      version: '2',
      display_name: 'Same Name',
      source: 'https://same.example.com',
      server_json: { name: 'test', version: '1.0', title: 'Same', description: 'Same desc' },
    });
    renderCompare({ baselineVersion: same, comparedVersion: sameV2 });
    expect(screen.getByText('Identical:')).toBeInTheDocument();
  });

  it('calls onSwitchSides when switch button is clicked', async () => {
    const onSwitchSides = jest.fn();
    renderCompare({ onSwitchSides });
    await userEvent.click(screen.getAllByRole('button', { name: /Switch sides/ })[0]);
    expect(onSwitchSides).toHaveBeenCalledTimes(1);
  });

  it('renders Empty fallback when baseline has no server_json', () => {
    const emptyVersion = createMockMCPServerVersion({
      version: '0',
      server_json: undefined as any,
      tools: [],
    });
    renderCompare({ baselineVersion: emptyVersion });
    const empties = screen.getAllByText('Empty');
    expect(empties.length).toBeGreaterThanOrEqual(1);
  });

  it('renders metadata tags when present', () => {
    renderCompare();
    expect(screen.getByText('env')).toBeInTheDocument();
  });
});

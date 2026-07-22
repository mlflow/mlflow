import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { RemotesSubsection } from './RemotesSubsection';
import { PackagesSubsection } from './PackagesSubsection';
import { ServerJSONSection } from './ServerJSONSection';
import { createMockMCPServer, createMockMCPServerVersion } from '../test-utils';
import type { TransportType, ServerJSONPayload } from '../types';

// Mock useServerState to control permission flags
jest.mock('../hooks/useServerState', () => ({
  useServerState: jest.fn(),
}));

// Mock useConnectOptionToggle to control connect options
jest.mock('../hooks/useConnectOptionToggle', () => ({
  useConnectOptionToggle: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports -- lazy import so the mock is in place before the module initialises
const { useServerState } = require('../hooks/useServerState') as {
  useServerState: jest.Mock;
};

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { useConnectOptionToggle } = require('../hooks/useConnectOptionToggle') as {
  useConnectOptionToggle: jest.Mock;
};

const mockPermissions = ({
  canUpdate = false,
  showVisibilityControls = false,
}: {
  canUpdate?: boolean;
  showVisibilityControls?: boolean;
} = {}) => {
  useServerState.mockReturnValue({
    canUpdate,
    canDelete: false,
    canManage: false,
    isDimmed: false,
    isUnavailable: false,
    showVisibilityControls,
    isAuthAvailable: true,
  });
};

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

// ── RemotesSubsection ──

describe('RemotesSubsection', () => {
  const remoteA = { type: 'streamable-http' as TransportType, url: 'https://a.example.com/mcp' };
  const remoteB = { type: 'sse' as TransportType, url: 'https://b.example.com/mcp' };

  it('renders when there are visible remotes', () => {
    render(
      <Wrapper>
        <RemotesSubsection remotes={[remoteA, remoteB]} derivedName="test" />
      </Wrapper>,
    );
    expect(screen.getByText('https://a.example.com/mcp')).toBeInTheDocument();
    expect(screen.getByText('https://b.example.com/mcp')).toBeInTheDocument();
  });

  it('returns null when all remotes are hidden via connectOptions and showVisibilityControls is false', () => {
    const connectOptions = {
      'remote:https://a.example.com/mcp': { hidden: true },
      'remote:https://b.example.com/mcp': { hidden: true },
    };
    const { container } = render(
      <Wrapper>
        <RemotesSubsection
          remotes={[remoteA, remoteB]}
          derivedName="test"
          showVisibilityControls={false}
          connectOptions={connectOptions}
        />
      </Wrapper>,
    );
    expect(container.innerHTML).toBe('');
  });

  it('shows all remotes including hidden ones when showVisibilityControls is true', () => {
    const connectOptions = {
      'remote:https://a.example.com/mcp': { hidden: true },
      'remote:https://b.example.com/mcp': { hidden: true },
    };
    render(
      <Wrapper>
        <RemotesSubsection
          remotes={[remoteA, remoteB]}
          derivedName="test"
          showVisibilityControls
          connectOptions={connectOptions}
        />
      </Wrapper>,
    );
    expect(screen.getByText('https://a.example.com/mcp')).toBeInTheDocument();
    expect(screen.getByText('https://b.example.com/mcp')).toBeInTheDocument();
  });
});

// ── PackagesSubsection ──

describe('PackagesSubsection', () => {
  const pkgA = {
    registryType: 'npm',
    identifier: '@modelcontextprotocol/server-a',
    transport: { type: 'stdio' as TransportType },
  };
  const pkgB = {
    registryType: 'pip',
    identifier: 'mcp-server-b',
    transport: { type: 'stdio' as TransportType },
  };

  it('renders when there are visible packages', () => {
    render(
      <Wrapper>
        <PackagesSubsection packages={[pkgA, pkgB]} derivedName="test" />
      </Wrapper>,
    );
    expect(screen.getByText('@modelcontextprotocol/server-a')).toBeInTheDocument();
    expect(screen.getByText('mcp-server-b')).toBeInTheDocument();
  });

  it('returns null when all packages are hidden via connectOptions and showVisibilityControls is false', () => {
    const connectOptions = {
      'npm:@modelcontextprotocol/server-a': { hidden: true },
      'pip:mcp-server-b': { hidden: true },
    };
    const { container } = render(
      <Wrapper>
        <PackagesSubsection
          packages={[pkgA, pkgB]}
          derivedName="test"
          showVisibilityControls={false}
          connectOptions={connectOptions}
        />
      </Wrapper>,
    );
    expect(container.innerHTML).toBe('');
  });

  it('shows all packages including hidden ones when showVisibilityControls is true', () => {
    const connectOptions = {
      'npm:@modelcontextprotocol/server-a': { hidden: true },
      'pip:mcp-server-b': { hidden: true },
    };
    render(
      <Wrapper>
        <PackagesSubsection
          packages={[pkgA, pkgB]}
          derivedName="test"
          showVisibilityControls
          connectOptions={connectOptions}
        />
      </Wrapper>,
    );
    expect(screen.getByText('@modelcontextprotocol/server-a')).toBeInTheDocument();
    expect(screen.getByText('mcp-server-b')).toBeInTheDocument();
  });
});

// ── ServerJSONSection ──

describe('ServerJSONSection', () => {
  const serverJson: ServerJSONPayload = {
    name: 'io.github.test/server',
    version: '1.0.0',
    remotes: [{ type: 'streamable-http' as TransportType, url: 'https://remote.example.com/mcp' }],
    packages: [
      {
        registryType: 'npm',
        identifier: '@test/mcp-server',
        transport: { type: 'stdio' as TransportType },
      },
    ],
  };
  const server = createMockMCPServer({ name: 'io.github.test/server' });
  const version = createMockMCPServerVersion({ name: 'io.github.test/server' });

  beforeEach(() => {
    useConnectOptionToggle.mockReturnValue({
      connectOptions: undefined,
      handleToggleConnectOption: jest.fn(),
    });
  });

  it('shows raw server.json toggle when canUpdate is true', () => {
    mockPermissions({ canUpdate: true });
    render(
      <Wrapper>
        <ServerJSONSection serverJson={serverJson} server={server} version={version} />
      </Wrapper>,
    );
    expect(screen.getByText('View raw server.json')).toBeInTheDocument();
  });

  it('hides raw server.json toggle when canUpdate is false', () => {
    mockPermissions({ canUpdate: false });
    render(
      <Wrapper>
        <ServerJSONSection serverJson={serverJson} server={server} version={version} />
      </Wrapper>,
    );
    expect(screen.queryByText('View raw server.json')).not.toBeInTheDocument();
  });
});

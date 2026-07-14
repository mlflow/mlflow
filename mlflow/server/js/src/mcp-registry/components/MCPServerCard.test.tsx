import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { MCPServerCard } from './MCPServerCard';
import { createMockMCPServer } from '../test-utils';
import type { MCPServer } from '../types';

const renderCard = (server: MCPServer) =>
  render(
    <IntlProvider locale="en">
      <TestRouter
        routes={[
          testRoute(
            <DesignSystemProvider>
              <MCPServerCard server={server} />
            </DesignSystemProvider>,
            '/',
          ),
        ]}
      />
    </IntlProvider>,
  );

describe('MCPServerCard', () => {
  it('renders server name when no display_name is set', () => {
    renderCard(createMockMCPServer({ name: 'io.github.test/my-server' }));
    expect(screen.getByText('io.github.test/my-server')).toBeInTheDocument();
  });

  it('renders name even when display_name is set', () => {
    renderCard(createMockMCPServer({ name: 'io.github.test/raw', display_name: 'Pretty Name' }));
    expect(screen.getByText('io.github.test/raw')).toBeInTheDocument();
    expect(screen.queryByText('Pretty Name')).not.toBeInTheDocument();
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
});

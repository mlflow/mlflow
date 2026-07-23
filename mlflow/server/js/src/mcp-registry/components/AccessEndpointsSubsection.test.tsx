import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { AccessEndpointsSubsection } from './AccessEndpointsSubsection';
import { createMockAccessEndpoint } from '../test-utils';
import { TransportType } from '../types';

// Mock useServerState to control permission flags without needing react-query / auth
jest.mock('../hooks/useServerState', () => ({
  useServerState: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports -- lazy import so the mock is in place before the module initialises
const { useServerState } = require('../hooks/useServerState') as {
  useServerState: jest.Mock;
};

const mockPermissions = ({
  canUpdate = false,
  canDelete = false,
  canManage = false,
}: {
  canUpdate?: boolean;
  canDelete?: boolean;
  canManage?: boolean;
} = {}) => {
  useServerState.mockReturnValue({
    canUpdate,
    canDelete,
    canManage,
    isDimmed: false,
    isUnavailable: false,
    showVisibilityControls: false,
    isAuthAvailable: true,
  });
};

const renderSubsection = (props: Partial<React.ComponentProps<typeof AccessEndpointsSubsection>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AccessEndpointsSubsection endpoints={[]} derivedName="Test Server" {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('AccessEndpointsSubsection', () => {
  beforeEach(() => {
    mockPermissions();
  });

  it('renders empty state when no endpoints are provided', () => {
    renderSubsection();
    expect(screen.getByText('No access endpoints configured for this server.')).toBeInTheDocument();
  });

  it('renders endpoints list with expandable rows', () => {
    const endpoints = [
      createMockAccessEndpoint({
        id: 'ae-1',
        url: 'https://example.com/mcp-a',
        transport_type: TransportType.STREAMABLE_HTTP,
      }),
      createMockAccessEndpoint({
        id: 'ae-2',
        url: 'https://example.com/mcp-b',
        transport_type: TransportType.SSE,
      }),
    ];
    renderSubsection({ endpoints });

    expect(screen.getByText('https://example.com/mcp-a')).toBeInTheDocument();
    expect(screen.getByText('https://example.com/mcp-b')).toBeInTheDocument();
    expect(screen.getByText('streamable-http')).toBeInTheDocument();
    expect(screen.getByText('sse')).toBeInTheDocument();
  });

  it('shows Add endpoint button when canUpdate is true and onAddEndpoint is provided', () => {
    mockPermissions({ canUpdate: true });
    renderSubsection({ onAddEndpoint: jest.fn() });
    expect(screen.getByText('Add endpoint')).toBeInTheDocument();
  });

  it('hides Add endpoint button when canUpdate is false', () => {
    mockPermissions({ canUpdate: false });
    renderSubsection({ onAddEndpoint: jest.fn() });
    expect(screen.queryByText('Add endpoint')).not.toBeInTheDocument();
  });

  it('hides Add endpoint button when onAddEndpoint is not provided', () => {
    mockPermissions({ canUpdate: true });
    renderSubsection({ onAddEndpoint: undefined });
    expect(screen.queryByText('Add endpoint')).not.toBeInTheDocument();
  });

  it('shows edit and delete icons per row when canUpdate and canDelete are true', () => {
    mockPermissions({ canUpdate: true, canDelete: true });
    const endpoints = [createMockAccessEndpoint({ id: 'ae-1' })];
    renderSubsection({
      endpoints,
      onEditEndpoint: jest.fn(),
      onDeleteEndpoint: jest.fn(),
    });

    expect(screen.getByLabelText('Edit access endpoint')).toBeInTheDocument();
    expect(screen.getByLabelText('Delete access endpoint')).toBeInTheDocument();
  });

  it('hides edit icon when canUpdate is false', () => {
    mockPermissions({ canUpdate: false, canDelete: true });
    const endpoints = [createMockAccessEndpoint({ id: 'ae-1' })];
    renderSubsection({
      endpoints,
      onEditEndpoint: jest.fn(),
      onDeleteEndpoint: jest.fn(),
    });

    expect(screen.queryByLabelText('Edit access endpoint')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete access endpoint')).toBeInTheDocument();
  });

  it('hides delete icon when canDelete is false', () => {
    mockPermissions({ canUpdate: true, canDelete: false });
    const endpoints = [createMockAccessEndpoint({ id: 'ae-1' })];
    renderSubsection({
      endpoints,
      onEditEndpoint: jest.fn(),
      onDeleteEndpoint: jest.fn(),
    });

    expect(screen.getByLabelText('Edit access endpoint')).toBeInTheDocument();
    expect(screen.queryByLabelText('Delete access endpoint')).not.toBeInTheDocument();
  });

  it('hides both action icons when neither callback is provided', () => {
    mockPermissions({ canUpdate: true, canDelete: true });
    const endpoints = [createMockAccessEndpoint({ id: 'ae-1' })];
    renderSubsection({ endpoints });

    expect(screen.queryByLabelText('Edit access endpoint')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Delete access endpoint')).not.toBeInTheDocument();
  });

  it('expanding a row shows connection instructions content', async () => {
    mockPermissions();
    const endpoints = [
      createMockAccessEndpoint({
        id: 'ae-1',
        url: 'https://example.com/mcp',
        transport_type: TransportType.STREAMABLE_HTTP,
      }),
    ];
    renderSubsection({ endpoints });

    // Click the expandable row to expand it
    const expandButton = screen.getByLabelText('Expand endpoint https://example.com/mcp');
    await userEvent.click(expandButton);

    // The expanded content should show "Target:" label
    expect(screen.getByText('Target:')).toBeInTheDocument();
  });

  it('calls onAddEndpoint when Add endpoint button is clicked', async () => {
    mockPermissions({ canUpdate: true });
    const onAddEndpoint = jest.fn();
    renderSubsection({ onAddEndpoint });

    await userEvent.click(screen.getByText('Add endpoint'));
    expect(onAddEndpoint).toHaveBeenCalledTimes(1);
  });

  it('renders the section heading', () => {
    renderSubsection();
    expect(screen.getByText('Access endpoints')).toBeInTheDocument();
  });
});

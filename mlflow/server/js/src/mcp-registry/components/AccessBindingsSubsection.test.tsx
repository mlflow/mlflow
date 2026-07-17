import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { AccessBindingsSubsection } from './AccessBindingsSubsection';
import { createMockAccessBinding } from '../test-utils';
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

const renderSubsection = (props: Partial<React.ComponentProps<typeof AccessBindingsSubsection>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AccessBindingsSubsection bindings={[]} derivedName="Test Server" {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('AccessBindingsSubsection', () => {
  beforeEach(() => {
    mockPermissions();
  });

  it('renders empty state when no bindings are provided', () => {
    renderSubsection();
    expect(screen.getByText('No access endpoints configured for this server.')).toBeInTheDocument();
  });

  it('renders bindings list with expandable rows', () => {
    const bindings = [
      createMockAccessBinding({
        binding_id: 1,
        endpoint_url: 'https://example.com/mcp-a',
        transport_type: TransportType.STREAMABLE_HTTP,
      }),
      createMockAccessBinding({
        binding_id: 2,
        endpoint_url: 'https://example.com/mcp-b',
        transport_type: TransportType.SSE,
      }),
    ];
    renderSubsection({ bindings });

    expect(screen.getByText('https://example.com/mcp-a')).toBeInTheDocument();
    expect(screen.getByText('https://example.com/mcp-b')).toBeInTheDocument();
    expect(screen.getByText('streamable-http')).toBeInTheDocument();
    expect(screen.getByText('sse')).toBeInTheDocument();
  });

  it('shows Add endpoint button when canUpdate is true and onAddBinding is provided', () => {
    mockPermissions({ canUpdate: true });
    renderSubsection({ onAddBinding: jest.fn() });
    expect(screen.getByText('Add endpoint')).toBeInTheDocument();
  });

  it('hides Add endpoint button when canUpdate is false', () => {
    mockPermissions({ canUpdate: false });
    renderSubsection({ onAddBinding: jest.fn() });
    expect(screen.queryByText('Add endpoint')).not.toBeInTheDocument();
  });

  it('hides Add endpoint button when onAddBinding is not provided', () => {
    mockPermissions({ canUpdate: true });
    renderSubsection({ onAddBinding: undefined });
    expect(screen.queryByText('Add endpoint')).not.toBeInTheDocument();
  });

  it('shows edit and delete icons per row when canUpdate and canDelete are true', () => {
    mockPermissions({ canUpdate: true, canDelete: true });
    const bindings = [createMockAccessBinding({ binding_id: 1 })];
    renderSubsection({
      bindings,
      onEditBinding: jest.fn(),
      onDeleteBinding: jest.fn(),
    });

    expect(screen.getByLabelText('Edit access endpoint')).toBeInTheDocument();
    expect(screen.getByLabelText('Delete access endpoint')).toBeInTheDocument();
  });

  it('hides edit icon when canUpdate is false', () => {
    mockPermissions({ canUpdate: false, canDelete: true });
    const bindings = [createMockAccessBinding({ binding_id: 1 })];
    renderSubsection({
      bindings,
      onEditBinding: jest.fn(),
      onDeleteBinding: jest.fn(),
    });

    expect(screen.queryByLabelText('Edit access endpoint')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete access endpoint')).toBeInTheDocument();
  });

  it('hides delete icon when canDelete is false', () => {
    mockPermissions({ canUpdate: true, canDelete: false });
    const bindings = [createMockAccessBinding({ binding_id: 1 })];
    renderSubsection({
      bindings,
      onEditBinding: jest.fn(),
      onDeleteBinding: jest.fn(),
    });

    expect(screen.getByLabelText('Edit access endpoint')).toBeInTheDocument();
    expect(screen.queryByLabelText('Delete access endpoint')).not.toBeInTheDocument();
  });

  it('hides both action icons when neither callback is provided', () => {
    mockPermissions({ canUpdate: true, canDelete: true });
    const bindings = [createMockAccessBinding({ binding_id: 1 })];
    renderSubsection({ bindings });

    expect(screen.queryByLabelText('Edit access endpoint')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Delete access endpoint')).not.toBeInTheDocument();
  });

  it('expanding a row shows connection instructions content', async () => {
    mockPermissions();
    const bindings = [
      createMockAccessBinding({
        binding_id: 1,
        endpoint_url: 'https://example.com/mcp',
        transport_type: TransportType.STREAMABLE_HTTP,
      }),
    ];
    renderSubsection({ bindings });

    // Click the expandable row to expand it
    const expandButton = screen.getByLabelText('Expand binding https://example.com/mcp');
    await userEvent.click(expandButton);

    // The expanded content should show "Target:" label
    expect(screen.getByText('Target:')).toBeInTheDocument();
  });

  it('calls onAddBinding when Add endpoint button is clicked', async () => {
    mockPermissions({ canUpdate: true });
    const onAddBinding = jest.fn();
    renderSubsection({ onAddBinding });

    await userEvent.click(screen.getByText('Add endpoint'));
    expect(onAddBinding).toHaveBeenCalledTimes(1);
  });

  it('renders the section heading', () => {
    renderSubsection();
    expect(screen.getByText('Access endpoints')).toBeInTheDocument();
  });
});

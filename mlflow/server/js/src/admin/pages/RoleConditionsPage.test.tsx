import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import RoleConditionsPage from './RoleConditionsPage';

const mockAddCondition = jest.fn();
const mockRemoveCondition = jest.fn();
let mockConditions: any[] = [];
let mockConditionsError: Error | null = null;

jest.mock('../hooks', () => ({
  useRoleDetailQuery: () => ({
    data: { role: { id: 7, name: 'ml-engineer', workspace: 'default', description: null, permissions: [] } },
    isLoading: false,
    error: null,
  }),
  useRoleMutationConditionsQuery: () => ({
    data: { mutation_conditions: mockConditions },
    isLoading: false,
    error: mockConditionsError,
  }),
  useAddMutationCondition: () => ({ mutateAsync: mockAddCondition, isLoading: false }),
  useRemoveMutationCondition: () => ({ mutateAsync: mockRemoveCondition, isLoading: false }),
  useResourceOptionsQuery: () => ({ options: [], isLoading: false, error: null }),
  useWithSettingsReturnTo: () => (route: string) => route,
}));

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  useParams: () => ({ roleId: '7' }),
  Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>,
}));

const condition = (overrides: Record<string, unknown> = {}) => ({
  id: 1,
  role_id: 7,
  resource_type: 'run',
  condition_slot: 1,
  parent_resource_type: null,
  parent_resource_id: null,
  value_condition: null,
  target_condition: "tags.lifecycle != 'prod'",
  ...overrides,
});

describe('RoleConditionsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockConditions = [];
    mockConditionsError = null;
    mockAddCondition.mockResolvedValue({} as never);
    mockRemoveCondition.mockResolvedValue({} as never);
  });

  it('states that conditions subtract and stack', () => {
    // The single most expensive thing for an admin to get backwards. A condition
    // narrows a grant and cannot widen one, and a second role cannot lift the
    // first one's restriction -- so the page has to say so rather than leaving it
    // to be inferred from a table of filters.
    renderWithDesignSystem(<RoleConditionsPage />);
    expect(screen.getByText(/Conditions subtract, and they stack/)).toBeInTheDocument();
    expect(screen.getByText(/Adding a second role cannot lift a restriction/)).toBeInTheDocument();
  });

  it('says that admins bypass conditions', () => {
    // Otherwise an admin tests their own condition, sees it ignored, and concludes
    // the feature is broken.
    renderWithDesignSystem(<RoleConditionsPage />);
    expect(screen.getByText(/Admins and workspace managers bypass conditions/)).toBeInTheDocument();
  });

  it('renders an empty state that says grants apply in full', () => {
    renderWithDesignSystem(<RoleConditionsPage />);
    expect(screen.getByText('No conditions')).toBeInTheDocument();
    expect(screen.getByText(/grants apply in full/)).toBeInTheDocument();
  });

  it('shows each stored condition with its filters and scope', () => {
    mockConditions = [
      condition(),
      condition({
        id: 2,
        resource_type: 'trace',
        parent_resource_type: 'experiment',
        parent_resource_id: '42',
        value_condition: "tag_value != 'prod'",
        target_condition: null,
      }),
    ];
    renderWithDesignSystem(<RoleConditionsPage />);
    expect(screen.getByText("tags.lifecycle != 'prod'")).toBeInTheDocument();
    expect(screen.getByText("tag_value != 'prod'")).toBeInTheDocument();
    expect(screen.getByText('All in workspace')).toBeInTheDocument();
    expect(screen.getByText('Experiment 42')).toBeInTheDocument();
  });

  it('keeps Add disabled until a filter is entered', async () => {
    renderWithDesignSystem(<RoleConditionsPage />);
    const addButton = screen.getByRole('button', { name: 'Add condition' });
    expect(addButton).toBeDisabled();

    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.x = 'y'");
    await waitFor(() => expect(addButton).toBeEnabled());
  });

  it('sends an empty filter as null rather than an empty string', async () => {
    // An empty string reaches the condition parser and is reported as a syntax
    // error; null is how the server is told the filter is absent.
    renderWithDesignSystem(<RoleConditionsPage />);
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), "tags.x = 'y'");
    await userEvent.click(screen.getByRole('button', { name: 'Add condition' }));

    await waitFor(() => expect(mockAddCondition).toHaveBeenCalledTimes(1));
    expect(mockAddCondition).toHaveBeenCalledWith(
      expect.objectContaining({
        role_id: 7,
        resource_type: 'experiment',
        value_condition: null,
        target_condition: "tags.x = 'y'",
        parent_resource_type: null,
        parent_resource_id: null,
      }),
    );
  });

  it('surfaces the server message when an add is rejected', async () => {
    // A rejected filter string comes back with the parse error, which is the only
    // thing that tells the admin how to fix it.
    mockAddCondition.mockRejectedValue(new Error("Invalid comparator '>' for tag condition") as never);
    renderWithDesignSystem(<RoleConditionsPage />);
    await userEvent.type(screen.getByPlaceholderText("tags.lifecycle != 'prod'"), 'tags.x > 1');
    await userEvent.click(screen.getByRole('button', { name: 'Add condition' }));

    await waitFor(() =>
      expect(screen.getByText("Invalid comparator '>' for tag condition")).toBeInTheDocument(),
    );
  });

  it('removes a condition by its own id', async () => {
    // Conditions are addressed by condition id, not by (role, resource_type) --
    // a role may hold many of one type, so the pair does not identify one.
    mockConditions = [condition({ id: 99 })];
    renderWithDesignSystem(<RoleConditionsPage />);
    await userEvent.click(screen.getByRole('button', { name: /Remove Run condition/ }));
    await waitFor(() => expect(mockRemoveCondition).toHaveBeenCalledWith(99));
  });

  it('reports a failure to load conditions instead of rendering an empty list', () => {
    // An empty table and a failed fetch look identical, and they mean opposite
    // things: "nothing restricts this role" versus "we do not know".
    mockConditionsError = new Error('boom');
    renderWithDesignSystem(<RoleConditionsPage />);
    expect(screen.getByText('Failed to load conditions')).toBeInTheDocument();
  });
});

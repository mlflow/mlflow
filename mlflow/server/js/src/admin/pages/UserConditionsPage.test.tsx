import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import UserConditionsPage from './UserConditionsPage';

let mockGroups: any[] = [];
let mockTotal = 0;
let mockError: Error | null = null;

jest.mock('../hooks', () => ({
  useUserMutationConditionsQuery: () => ({
    groups: mockGroups,
    isLoading: false,
    error: mockError,
    totalConditions: mockTotal,
  }),
  useWithSettingsReturnTo: () => (route: string) => route,
}));

// Drags in its own query stack and is not what these cases are about.
jest.mock('../components/EditAccessModal', () => ({
  EditAccessModal: () => null,
}));

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  useParams: () => ({ username: 'alice' }),
  Link: ({ children }: { children: React.ReactNode }) => <a>{children}</a>,
}));

const group = (roleName: string, roleId: number, conditions: any[]) => ({
  role: { id: roleId, name: roleName, workspace: 'default', description: null, permissions: [] },
  conditions,
  isLoading: false,
  error: null,
});

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

describe('UserConditionsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGroups = [];
    mockTotal = 0;
    mockError = null;
  });

  it('warns that assigning another role cannot lift a condition', () => {
    // This page carries a "Manage roles" button, so it is exactly where someone
    // would try to fix a refused write by adding a role. Conditions AND across
    // every role held, so that makes the restriction worse, not better.
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText(/Adding a role cannot lift a condition/)).toBeInTheDocument();
    expect(screen.getByText(/adds its grants but also adds its conditions/)).toBeInTheDocument();
  });

  it('offers role management, since that is the part that belongs to the user', () => {
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByRole('button', { name: 'Manage roles' })).toBeInTheDocument();
  });

  it('groups conditions by the role that carries them', () => {
    // "Which role is doing this to me?" is the question here: the role is the only
    // place the restriction can be changed.
    mockGroups = [
      group('ml-engineer', 7, [condition()]),
      group('auditor', 8, [condition({ id: 2, resource_type: 'trace' })]),
    ];
    mockTotal = 2;
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText('ml-engineer')).toBeInTheDocument();
    expect(screen.getByText('auditor')).toBeInTheDocument();
  });

  it('hides roles that carry no condition', () => {
    // Listing every role would bury the few that actually restrict anything.
    mockGroups = [group('ml-engineer', 7, [condition()]), group('unrestricted', 9, [])];
    mockTotal = 1;
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText('ml-engineer')).toBeInTheDocument();
    expect(screen.queryByText('unrestricted')).not.toBeInTheDocument();
  });

  it('names a synthetic role as Direct grants without hiding its conditions', () => {
    // A direct grant is backed by a ``__user_<id>__`` role. Leaking that name is
    // noise, but dropping the row would hide a live restriction.
    mockGroups = [group('__user_1__', 19, [condition({ target_condition: "tags.env = 'dev'" })])];
    mockTotal = 1;
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText('Direct grants')).toBeInTheDocument();
    expect(screen.queryByText('__user_1__')).not.toBeInTheDocument();
    expect(screen.getByText("tags.env = 'dev'")).toBeInTheDocument();
  });

  it('distinguishes holding no roles from holding roles that restrict nothing', () => {
    mockGroups = [];
    mockTotal = 0;
    const { unmount } = renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText(/holds no roles/)).toBeInTheDocument();
    unmount();

    mockGroups = [group('unrestricted', 9, [])];
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText(/None of this user's roles carry a condition/)).toBeInTheDocument();
  });

  it('reports a fetch failure instead of claiming there are no conditions', () => {
    mockError = new Error('boom');
    renderWithDesignSystem(<UserConditionsPage />);
    expect(screen.getByText('Failed to load conditions')).toBeInTheDocument();
  });
});

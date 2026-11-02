import { useIntl } from 'react-intl';
import { getWorkspacesEnabledSync } from '../../../experiment-tracking/hooks/useServerInfo';
import type { TargetScope } from '../../types';

export type UnscopedTargetScope = Extract<TargetScope, 'GLOBAL' | 'WORKSPACE'>;

/**
 * Labels for the budget scope selector, shared with the "Applies to" column so
 * the two can't drift apart. GLOBAL and WORKSPACE policies both apply to every
 * endpoint and user, but with workspaces enabled GLOBAL covers all workspaces
 * and WORKSPACE only its own, so the label names which one it is rather than
 * claiming to cover everything.
 */
export const useBudgetScopeLabels = () => {
  const intl = useIntl();

  const allEndpointsAndUsers = intl.formatMessage({
    defaultMessage: 'All endpoints and users',
    description: 'Budget scope label covering every endpoint and user',
  });
  const all: Record<UnscopedTargetScope, string> = getWorkspacesEnabledSync()
    ? {
        GLOBAL: intl.formatMessage({
          defaultMessage: 'All endpoints and users in all workspaces',
          description: 'Budget scope label covering every endpoint and user across every workspace',
        }),
        WORKSPACE: intl.formatMessage({
          defaultMessage: 'All endpoints and users in this workspace',
          description: 'Budget scope label covering every endpoint and user within the current workspace',
        }),
      }
    : { GLOBAL: allEndpointsAndUsers, WORKSPACE: allEndpointsAndUsers };

  return {
    all,
    endpoint: intl.formatMessage({
      defaultMessage: 'Specific endpoint',
      description: 'Budget scope option limiting the policy to one endpoint',
    }),
    user: intl.formatMessage({
      defaultMessage: 'Specific user',
      description: 'Budget scope option limiting the policy to one user',
    }),
  };
};

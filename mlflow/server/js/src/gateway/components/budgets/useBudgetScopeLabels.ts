import { useIntl } from 'react-intl';
import { getWorkspacesEnabledSync } from '../../../experiment-tracking/hooks/useServerInfo';

/**
 * Labels for the budget scope selector, shared with the "Applies to" column so
 * the two can't drift apart. The unscoped choice maps to WORKSPACE when
 * workspaces are enabled and GLOBAL otherwise, so its label is narrowed to the
 * current workspace in the former case rather than claiming to cover everything.
 */
export const useBudgetScopeLabels = () => {
  const intl = useIntl();

  const all = getWorkspacesEnabledSync()
    ? intl.formatMessage({
        defaultMessage: 'All endpoints and users in this workspace',
        description: 'Budget scope label covering every endpoint and user within the current workspace',
      })
    : intl.formatMessage({
        defaultMessage: 'All endpoints and users',
        description: 'Budget scope label covering every endpoint and user',
      });

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

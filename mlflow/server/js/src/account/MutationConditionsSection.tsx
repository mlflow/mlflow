import { useMemo } from 'react';
import { ConditionsTable } from '../admin/components/ConditionsTable';
import type { UserRoleConditionRow } from './types';
import { isSyntheticUserRole } from './types';

interface Props {
  /** Every condition on every role the user holds, from the self endpoint. */
  conditions: UserRoleConditionRow[];
  isLoading?: boolean;
  /** Surfaced by the table itself: a failed fetch and an empty list mean opposite things. */
  error?: unknown;
}

/**
 * The mutation conditions restricting the signed-in user.
 *
 * Renders through the admin UI's shared ``ConditionsTable`` rather than a parallel table,
 * so a user sees their conditions in exactly the columns an admin sees them in. A second
 * implementation would be free to drift, and the two views describe the same policy.
 *
 * The carrying role is a column rather than a grouping, matching the admin's per-user tab.
 */
export const MutationConditionsSection = ({ conditions, isLoading, error }: Props) => {
  // ``rowSuffix`` receives a bare condition, which carries ``role_id`` but not the role
  // name, so the mapping is built here -- the same shape the admin's user tab uses.
  const roleNameById = useMemo(() => {
    const byId = new Map<number, string>();
    for (const condition of conditions) {
      // A condition attached directly to the user sits on the synthetic
      // ``__user_<id>__`` role. Name it for what it is rather than leaking the
      // internal name.
      byId.set(condition.role_id, isSyntheticUserRole(condition.role_name) ? 'Direct grants' : condition.role_name);
    }
    return byId;
  }, [conditions]);

  return (
    <ConditionsTable
      conditions={conditions}
      isLoading={isLoading}
      error={error}
      emptyDescription="None of your roles carry a mutation condition."
      suffixHeader="From Role"
      rowSuffix={(condition) => roleNameById.get(condition.role_id)}
    />
  );
};

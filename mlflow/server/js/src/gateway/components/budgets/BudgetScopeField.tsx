import {
  FormUI,
  Input,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useEndpointsQuery } from '../../hooks/useEndpointsQuery';
import type { TargetScope } from '../../types';
import { getWorkspacesEnabledSync } from '../../../experiment-tracking/hooks/useServerInfo';
import { useBudgetScopeLabels, type UnscopedTargetScope } from './useBudgetScopeLabels';

type BudgetScopeChoice = 'ALL' | 'ENDPOINT' | 'USER';

export interface BudgetScopeFormValue {
  scope: BudgetScopeChoice;
  endpointId: string;
  username: string;
}

/**
 * The scope that the "All" choice saves as. The selector shows GLOBAL and
 * WORKSPACE as one choice, so a policy that already has one of them keeps it:
 * re-deriving it would silently narrow a GLOBAL policy to a single workspace on
 * a workspace-enabled server.
 */
const getUnscopedTargetScope = (existingScope?: TargetScope): UnscopedTargetScope => {
  if (existingScope === 'GLOBAL' || existingScope === 'WORKSPACE') {
    return existingScope;
  }
  return getWorkspacesEnabledSync() ? 'WORKSPACE' : 'GLOBAL';
};

export const getBudgetScopePayload = (
  { scope, endpointId, username }: BudgetScopeFormValue,
  existingScope?: TargetScope,
): { target_scope: TargetScope; target_value?: string } => {
  if (scope === 'ENDPOINT') {
    return { target_scope: 'ENDPOINT', target_value: endpointId };
  }
  if (scope === 'USER') {
    return { target_scope: 'USER', target_value: username.trim() };
  }
  return { target_scope: getUnscopedTargetScope(existingScope) };
};

export const useBudgetScopeField = ({ scope, endpointId, username }: BudgetScopeFormValue) => {
  const { data: endpoints, isLoading, error } = useEndpointsQuery();
  // `useEndpointsQuery` yields an empty array both while loading and on failure,
  // so only a list we actually have says anything about which endpoints exist.
  const hasEndpointList = !isLoading && !error;
  // A policy can name an endpoint the list lacks: deleting an endpoint doesn't
  // cascade to budget policies, and with auth enabled the list omits endpoints
  // the caller can't read. The picker can't render a selection it has no option
  // for, so the id has to count as no selection at all — otherwise Save looks
  // available and fails server-side.
  const isEndpointMissing =
    scope === 'ENDPOINT' &&
    Boolean(endpointId) &&
    hasEndpointList &&
    !endpoints.some((endpoint) => endpoint.endpoint_id === endpointId);
  const isValid =
    scope === 'ENDPOINT' ? Boolean(endpointId) && !isEndpointMissing : scope !== 'USER' || username.trim().length > 0;

  return { endpoints, hasEndpointList, isEndpointMissing, isValid };
};

interface BudgetScopeFieldProps {
  componentId: string;
  value: BudgetScopeFormValue;
  onChange: (change: Partial<BudgetScopeFormValue>) => void;
  field: ReturnType<typeof useBudgetScopeField>;
  /** Scope of the policy being edited, if any. */
  existingScope?: TargetScope;
}

/** The "Applies to" section of the create and edit budget policy forms. */
export const BudgetScopeField = ({ componentId, value, onChange, field, existingScope }: BudgetScopeFieldProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const scopeLabels = useBudgetScopeLabels();
  const allLabel = scopeLabels.all[getUnscopedTargetScope(existingScope)];
  const { endpoints, hasEndpointList, isEndpointMissing } = field;
  const hasNoEndpoints = hasEndpointList && !endpoints.length;
  const scopeLabel = intl.formatMessage({ defaultMessage: 'Applies to', description: 'Budget scope label' });

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text bold>{scopeLabel}</Typography.Text>
      <SimpleSelect
        id={`${componentId}.scope`}
        componentId={`${componentId}.scope`}
        label={scopeLabel}
        value={value.scope}
        onChange={({ target }) => onChange({ scope: target.value as BudgetScopeChoice })}
      >
        <SimpleSelectOption value="ALL">{allLabel}</SimpleSelectOption>
        <SimpleSelectOption value="ENDPOINT">{scopeLabels.endpoint}</SimpleSelectOption>
        <SimpleSelectOption value="USER">{scopeLabels.user}</SimpleSelectOption>
      </SimpleSelect>
      {value.scope === 'ENDPOINT' && (
        <>
          <SimpleSelect
            id={`${componentId}.endpoint`}
            componentId={`${componentId}.endpoint`}
            label={intl.formatMessage({
              defaultMessage: 'Endpoint',
              description: 'Accessible label for the budget policy endpoint selector',
            })}
            value={value.endpointId}
            onChange={({ target }) => onChange({ endpointId: target.value })}
            disabled={hasNoEndpoints}
            validationState={isEndpointMissing ? 'error' : undefined}
            placeholder={
              hasNoEndpoints
                ? intl.formatMessage({
                    defaultMessage: 'No endpoints available',
                    description: 'Placeholder for the budget policy endpoint selector when no endpoints exist',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Select an endpoint',
                    description: 'Placeholder for budget policy endpoint selector',
                  })
            }
          >
            {endpoints.map((endpoint) => (
              <SimpleSelectOption key={endpoint.endpoint_id} value={endpoint.endpoint_id}>
                {endpoint.name}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
          {isEndpointMissing && (
            <FormUI.Message
              type="error"
              message={intl.formatMessage(
                {
                  defaultMessage:
                    'The endpoint this policy applies to ({endpointId}) was deleted, or you don\'t have access to it. Select another endpoint, or choose "{allLabel}".',
                  description:
                    'Validation error shown when a budget policy targets an endpoint that is missing from the endpoint list',
                },
                { endpointId: value.endpointId, allLabel },
              )}
            />
          )}
        </>
      )}
      {value.scope === 'USER' && (
        <>
          <Input
            componentId={`${componentId}.username`}
            value={value.username}
            onChange={(e) => onChange({ username: e.target.value })}
            placeholder={intl.formatMessage({
              defaultMessage: 'Username, e.g., alice',
              description: 'Budget username placeholder',
            })}
          />
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="The budget applies only to requests made by this authenticated user. Requires server authentication to be enabled."
              description="Helper text for per-user budget scope"
            />
          </Typography.Text>
        </>
      )}
    </div>
  );
};

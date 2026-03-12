import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  Alert,
  InfoSmallIcon,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useUpdateBudgetPolicy } from '../../hooks/useUpdateBudgetPolicy';
import type { BudgetPolicy, DurationUnit, BudgetAction } from '../../types';
import { getWorkspacesEnabledSync } from '../../../experiment-tracking/hooks/useServerInfo';

type DurationPreset = 'DAILY' | 'WEEKLY' | 'MONTHLY';

const DURATION_MAP: Record<DurationPreset, { unit: DurationUnit; value: number }> = {
  DAILY: { unit: 'DAYS', value: 1 },
  WEEKLY: { unit: 'WEEKS', value: 1 },
  MONTHLY: { unit: 'MONTHS', value: 1 },
};

const toDurationPreset = (unit: DurationUnit, value: number): DurationPreset => {
  if (unit === 'DAYS' && value === 1) return 'DAILY';
  if (unit === 'WEEKS' && value === 1) return 'WEEKLY';
  if (unit === 'MONTHS' && value === 1) return 'MONTHLY';
  return 'MONTHLY';
};

interface EditBudgetPolicyModalProps {
  open: boolean;
  policy: BudgetPolicy | null;
  onClose: () => void;
  onSuccess?: () => void;
}

interface FormData {
  budgetAmount: string;
  duration: DurationPreset;
  budgetAction: BudgetAction;
}

export const EditBudgetPolicyModal = ({ open, policy, onClose, onSuccess }: EditBudgetPolicyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<FormData>({
    budgetAmount: '',
    duration: 'MONTHLY',
    budgetAction: 'REJECT',
  });
  const {
    mutateAsync: updateBudgetPolicy,
    isLoading,
    error: mutationError,
    reset: resetMutation,
  } = useUpdateBudgetPolicy();

  useEffect(() => {
    if (policy) {
      setFormData({
        budgetAmount: String(policy.budget_amount),
        duration: toDurationPreset(policy.duration_unit, policy.duration_value),
        budgetAction: policy.budget_action,
      });
      resetMutation();
    }
  }, [policy, resetMutation]);

  const handleClose = useCallback(() => {
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const handleFieldChange = useCallback(
    <K extends keyof FormData>(field: K, value: FormData[K]) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
      resetMutation();
    },
    [resetMutation],
  );

  const isFormValid = useMemo(() => {
    const amount = parseFloat(formData.budgetAmount);
    return !isNaN(amount) && amount > 0;
  }, [formData.budgetAmount]);

  const handleSubmit = useCallback(async () => {
    if (!isFormValid || !policy) return;

    const { unit, value } = DURATION_MAP[formData.duration];

    await updateBudgetPolicy({
      budget_policy_id: policy.budget_policy_id,
      budget_unit: 'USD',
      budget_amount: parseFloat(formData.budgetAmount),
      duration_unit: unit,
      duration_value: value,
      target_scope: getWorkspacesEnabledSync() ? 'WORKSPACE' : 'GLOBAL',
      budget_action: formData.budgetAction,
    }).then(() => {
      handleClose();
      onSuccess?.();
    });
  }, [isFormValid, policy, formData, updateBudgetPolicy, handleClose, onSuccess]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    if (message.length > 200) {
      return intl.formatMessage({
        defaultMessage: 'An error occurred while updating the budget policy. Please try again.',
        description: 'Generic error message for budget policy update',
      });
    }

    return message;
  }, [mutationError, intl]);

  if (!policy) return null;

  return (
    <Modal
      componentId="mlflow.gateway.edit-budget-policy-modal"
      title={intl.formatMessage({
        defaultMessage: 'Edit Budget Policy',
        description: 'Title for edit budget policy modal',
      })}
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Save Changes',
        description: 'Save changes button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel button text',
      })}
      confirmLoading={isLoading}
      okButtonProps={{ disabled: !isFormValid }}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {errorMessage && (
          <Alert
            componentId="mlflow.gateway.edit-budget-policy-modal.error"
            type="error"
            message={errorMessage}
            closable={false}
          />
        )}

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Budget amount (USD)" description="Budget amount label" />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.edit-budget-policy-modal.budget-amount"
            value={formData.budgetAmount}
            onChange={(e) => handleFieldChange('budgetAmount', e.target.value)}
            type="number"
            min={0}
            step="0.01"
          />
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Reset period" description="Budget reset period label" />
            </Typography.Text>
            <Tooltip
              componentId="mlflow.gateway.edit-budget-policy-modal.reset-period-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="<link>Learn more about budget time windows.</link>"
                  description="Tooltip with link to budget time windows documentation"
                  values={{
                    link: (chunks: React.ReactNode) => (
                      <a
                        href="https://mlflow.org/docs/latest/genai/governance/ai-gateway/budget-alerts-limits#time-windows"
                        target="_blank"
                        rel="noopener noreferrer"
                        css={{ color: 'inherit', textDecoration: 'underline' }}
                      >
                        {chunks}
                      </a>
                    ),
                  }}
                />
              }
            >
              <InfoSmallIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
            </Tooltip>
          </div>
          <SimpleSelect
            id="edit-budget-policy-duration"
            componentId="mlflow.gateway.edit-budget-policy-modal.duration"
            value={formData.duration}
            onChange={({ target }) => handleFieldChange('duration', target.value as DurationPreset)}
          >
            <SimpleSelectOption value="DAILY">Daily</SimpleSelectOption>
            <SimpleSelectOption value="WEEKLY">Weekly</SimpleSelectOption>
            <SimpleSelectOption value="MONTHLY">Monthly</SimpleSelectOption>
          </SimpleSelect>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="On exceeded" description="Budget on exceeded label" />
            </Typography.Text>
            <Tooltip
              componentId="mlflow.gateway.edit-budget-policy-modal.on-exceeded-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="Alert sends a webhook notification. <link>Learn how to set up webhooks.</link>"
                  description="Tooltip explaining budget exceeded actions and webhook setup"
                  values={{
                    link: (chunks: React.ReactNode) => (
                      <a
                        href="https://mlflow.org/docs/latest/ml/webhooks/"
                        target="_blank"
                        rel="noopener noreferrer"
                        css={{ color: 'inherit', textDecoration: 'underline' }}
                      >
                        {chunks}
                      </a>
                    ),
                  }}
                />
              }
            >
              <InfoSmallIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
            </Tooltip>
          </div>
          <SimpleSelect
            id="edit-budget-policy-on-exceeded"
            componentId="mlflow.gateway.edit-budget-policy-modal.on-exceeded"
            value={formData.budgetAction}
            onChange={({ target }) => handleFieldChange('budgetAction', target.value as BudgetAction)}
          >
            <SimpleSelectOption value="ALERT">Alert</SimpleSelectOption>
            <SimpleSelectOption value="REJECT">Reject requests</SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>
    </Modal>
  );
};

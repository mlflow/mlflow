import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  Alert,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useUpdateBudgetPolicy } from '../../hooks/useUpdateBudgetPolicy';
import type { BudgetPolicy, DurationUnit, TargetType, BudgetAction } from '../../types';

interface EditBudgetPolicyModalProps {
  open: boolean;
  policy: BudgetPolicy | null;
  onClose: () => void;
  onSuccess?: () => void;
}

interface FormData {
  budgetAmount: string;
  durationUnit: DurationUnit;
  durationValue: string;
  targetType: TargetType;
  budgetAction: BudgetAction;
}

export const EditBudgetPolicyModal = ({ open, policy, onClose, onSuccess }: EditBudgetPolicyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<FormData>({
    budgetAmount: '',
    durationUnit: 'DAYS',
    durationValue: '30',
    targetType: 'GLOBAL',
    budgetAction: 'REJECT',
  });
  const { mutateAsync: updateBudgetPolicy, isLoading, error: mutationError, reset: resetMutation } =
    useUpdateBudgetPolicy();

  useEffect(() => {
    if (policy) {
      setFormData({
        budgetAmount: String(policy.budget_amount),
        durationUnit: policy.duration_unit,
        durationValue: String(policy.duration_value),
        targetType: policy.target_type,
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
    if (isNaN(amount) || amount <= 0) return false;
    const duration = parseInt(formData.durationValue, 10);
    if (isNaN(duration) || duration <= 0) return false;
    return true;
  }, [formData.budgetAmount, formData.durationValue]);

  const handleSubmit = useCallback(async () => {
    if (!isFormValid || !policy) return;

    await updateBudgetPolicy({
      budget_policy_id: policy.budget_policy_id,
      budget_unit: 'USD',
      budget_amount: parseFloat(formData.budgetAmount),
      duration_unit: formData.durationUnit,
      duration_value: parseInt(formData.durationValue, 10),
      target_type: formData.targetType,
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

        <div css={{ display: 'flex', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, flex: 1 }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Duration type" description="Budget duration type label" />
            </Typography.Text>
            <SimpleSelect
              id="edit-budget-policy-duration-type"
              componentId="mlflow.gateway.edit-budget-policy-modal.duration-type"
              value={formData.durationUnit}
              onChange={({ target }) => handleFieldChange('durationUnit', target.value as DurationUnit)}
            >
              <SimpleSelectOption value="MINUTES">Minutes</SimpleSelectOption>
              <SimpleSelectOption value="HOURS">Hours</SimpleSelectOption>
              <SimpleSelectOption value="DAYS">Days</SimpleSelectOption>
              <SimpleSelectOption value="MONTHS">Months</SimpleSelectOption>
            </SimpleSelect>
          </div>

          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, flex: 1 }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Duration value" description="Budget duration value label" />
            </Typography.Text>
            <Input
              componentId="mlflow.gateway.edit-budget-policy-modal.duration-value"
              value={formData.durationValue}
              onChange={(e) => handleFieldChange('durationValue', e.target.value)}
              type="number"
              min={1}
              step="1"
            />
          </div>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Scope" description="Budget target type label" />
          </Typography.Text>
          <SimpleSelect
            id="edit-budget-policy-target-type"
            componentId="mlflow.gateway.edit-budget-policy-modal.target-type"
            value={formData.targetType}
            onChange={({ target }) => handleFieldChange('targetType', target.value as TargetType)}
          >
            <SimpleSelectOption value="GLOBAL">Global</SimpleSelectOption>
            <SimpleSelectOption value="WORKSPACE">Workspace</SimpleSelectOption>
          </SimpleSelect>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="On exceeded" description="Budget on exceeded label" />
          </Typography.Text>
          <SimpleSelect
            id="edit-budget-policy-on-exceeded"
            componentId="mlflow.gateway.edit-budget-policy-modal.on-exceeded"
            value={formData.budgetAction}
            onChange={({ target }) => handleFieldChange('budgetAction', target.value as BudgetAction)}
          >
            <SimpleSelectOption value="ALERT">Alert only</SimpleSelectOption>
            <SimpleSelectOption value="REJECT">Reject requests</SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>
    </Modal>
  );
};

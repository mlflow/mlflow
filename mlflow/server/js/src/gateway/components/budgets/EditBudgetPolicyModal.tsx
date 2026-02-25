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
import type { BudgetPolicy, DurationType, TargetType, OnExceededAction } from '../../types';

interface EditBudgetPolicyModalProps {
  open: boolean;
  policy: BudgetPolicy | null;
  onClose: () => void;
  onSuccess?: () => void;
}

interface FormData {
  name: string;
  limitUsd: string;
  durationType: DurationType;
  durationValue: string;
  targetType: TargetType;
  onExceeded: OnExceededAction;
}

export const EditBudgetPolicyModal = ({ open, policy, onClose, onSuccess }: EditBudgetPolicyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<FormData>({
    name: '',
    limitUsd: '',
    durationType: 'DAYS',
    durationValue: '30',
    targetType: 'GLOBAL',
    onExceeded: 'ALERT_AND_REJECT',
  });
  const { mutateAsync: updateBudgetPolicy, isLoading, error: mutationError, reset: resetMutation } =
    useUpdateBudgetPolicy();

  useEffect(() => {
    if (policy) {
      setFormData({
        name: policy.name,
        limitUsd: String(policy.limit_usd),
        durationType: policy.duration_type,
        durationValue: String(policy.duration_value),
        targetType: policy.target_type,
        onExceeded: policy.on_exceeded,
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
    if (!formData.name.trim()) return false;
    const limit = parseFloat(formData.limitUsd);
    if (isNaN(limit) || limit <= 0) return false;
    const duration = parseInt(formData.durationValue, 10);
    if (isNaN(duration) || duration <= 0) return false;
    return true;
  }, [formData.name, formData.limitUsd, formData.durationValue]);

  const handleSubmit = useCallback(async () => {
    if (!isFormValid || !policy) return;

    await updateBudgetPolicy({
      budget_policy_id: policy.budget_policy_id,
      name: formData.name.trim(),
      limit_usd: parseFloat(formData.limitUsd),
      duration_type: formData.durationType,
      duration_value: parseInt(formData.durationValue, 10),
      target_type: formData.targetType,
      on_exceeded: formData.onExceeded,
    }).then(() => {
      handleClose();
      onSuccess?.();
    });
  }, [isFormValid, policy, formData, updateBudgetPolicy, handleClose, onSuccess]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    if (message.toLowerCase().includes('unique constraint') || message.toLowerCase().includes('duplicate')) {
      return intl.formatMessage({
        defaultMessage: 'A budget policy with this name already exists. Please choose a different name.',
        description: 'Error message for duplicate budget policy name',
      });
    }

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
            <FormattedMessage defaultMessage="Policy name" description="Budget policy name label" />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.edit-budget-policy-modal.name"
            value={formData.name}
            onChange={(e) => handleFieldChange('name', e.target.value)}
          />
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Limit (USD)" description="Budget limit label" />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.edit-budget-policy-modal.limit"
            value={formData.limitUsd}
            onChange={(e) => handleFieldChange('limitUsd', e.target.value)}
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
              value={formData.durationType}
              onChange={({ target }) => handleFieldChange('durationType', target.value as DurationType)}
            >
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
            value={formData.onExceeded}
            onChange={({ target }) => handleFieldChange('onExceeded', target.value as OnExceededAction)}
          >
            <SimpleSelectOption value="ALERT">Alert only</SimpleSelectOption>
            <SimpleSelectOption value="REJECT">Reject requests</SimpleSelectOption>
            <SimpleSelectOption value="ALERT_AND_REJECT">Alert and reject requests</SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>
    </Modal>
  );
};

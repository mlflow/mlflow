import { useState, useCallback, useMemo } from 'react';
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
import { useCreateBudgetPolicy } from '../../hooks/useCreateBudgetPolicy';
import type { DurationType, TargetType, OnExceededAction } from '../../types';

interface CreateBudgetPolicyModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

interface FormData {
  budgetAmount: string;
  durationType: DurationType;
  durationValue: string;
  targetType: TargetType;
  onExceeded: OnExceededAction;
}

const INITIAL_FORM_DATA: FormData = {
  budgetAmount: '',
  durationType: 'DAYS',
  durationValue: '30',
  targetType: 'GLOBAL',
  onExceeded: 'REJECT',
};

export const CreateBudgetPolicyModal = ({ open, onClose, onSuccess }: CreateBudgetPolicyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<FormData>(INITIAL_FORM_DATA);
  const { mutateAsync: createBudgetPolicy, isLoading, error: mutationError, reset: resetMutation } =
    useCreateBudgetPolicy();

  const handleClose = useCallback(() => {
    setFormData(INITIAL_FORM_DATA);
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
    if (!isFormValid) return;

    await createBudgetPolicy({
      budget_type: 'USD',
      budget_amount: parseFloat(formData.budgetAmount),
      duration_type: formData.durationType,
      duration_value: parseInt(formData.durationValue, 10),
      target_type: formData.targetType,
      on_exceeded: formData.onExceeded,
    }).then(() => {
      handleClose();
      onSuccess?.();
    });
  }, [isFormValid, formData, createBudgetPolicy, handleClose, onSuccess]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    if (message.length > 200) {
      return intl.formatMessage({
        defaultMessage: 'An error occurred while creating the budget policy. Please try again.',
        description: 'Generic error message for budget policy creation',
      });
    }

    return message;
  }, [mutationError, intl]);

  return (
    <Modal
      componentId="mlflow.gateway.create-budget-policy-modal"
      title={intl.formatMessage({
        defaultMessage: 'Create Budget Policy',
        description: 'Title for create budget policy modal',
      })}
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Create',
        description: 'Create budget policy button text',
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
            componentId="mlflow.gateway.create-budget-policy-modal.error"
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
            componentId="mlflow.gateway.create-budget-policy-modal.budget-amount"
            value={formData.budgetAmount}
            onChange={(e) => handleFieldChange('budgetAmount', e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., 100.00',
              description: 'Budget amount placeholder',
            })}
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
              id="create-budget-policy-duration-type"
              componentId="mlflow.gateway.create-budget-policy-modal.duration-type"
              value={formData.durationType}
              onChange={({ target }) => handleFieldChange('durationType', target.value as DurationType)}
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
              componentId="mlflow.gateway.create-budget-policy-modal.duration-value"
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
            id="create-budget-policy-target-type"
            componentId="mlflow.gateway.create-budget-policy-modal.target-type"
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
            id="create-budget-policy-on-exceeded"
            componentId="mlflow.gateway.create-budget-policy-modal.on-exceeded"
            value={formData.onExceeded}
            onChange={({ target }) => handleFieldChange('onExceeded', target.value as OnExceededAction)}
          >
            <SimpleSelectOption value="ALERT">Alert only</SimpleSelectOption>
            <SimpleSelectOption value="REJECT">Reject requests</SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>
    </Modal>
  );
};

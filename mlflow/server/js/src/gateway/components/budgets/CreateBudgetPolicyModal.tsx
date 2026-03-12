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
import type { BudgetAction, DurationUnit } from '../../types';
import { getWorkspacesEnabledSync } from '../../../experiment-tracking/hooks/useServerInfo';

type DurationPreset = 'DAILY' | 'WEEKLY' | 'MONTHLY';

const DURATION_MAP: Record<DurationPreset, { unit: DurationUnit; value: number }> = {
  DAILY: { unit: 'DAYS', value: 1 },
  WEEKLY: { unit: 'WEEKS', value: 1 },
  MONTHLY: { unit: 'MONTHS', value: 1 },
};

interface CreateBudgetPolicyModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

interface FormData {
  budgetAmount: string;
  duration: DurationPreset;
  budgetAction: BudgetAction;
}

const INITIAL_FORM_DATA: FormData = {
  budgetAmount: '',
  duration: 'MONTHLY',
  budgetAction: 'REJECT',
};

export const CreateBudgetPolicyModal = ({ open, onClose, onSuccess }: CreateBudgetPolicyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<FormData>(INITIAL_FORM_DATA);
  const {
    mutateAsync: createBudgetPolicy,
    isLoading,
    error: mutationError,
    reset: resetMutation,
  } = useCreateBudgetPolicy();

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
    return !isNaN(amount) && amount > 0;
  }, [formData.budgetAmount]);

  const handleSubmit = useCallback(async () => {
    if (!isFormValid) return;

    const { unit, value } = DURATION_MAP[formData.duration];

    await createBudgetPolicy({
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
          <div css={{ maxWidth: 200 }}>
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
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Reset period" description="Budget reset period label" />
            </Typography.Text>
            <Tooltip
              componentId="mlflow.gateway.create-budget-policy-modal.reset-period-tooltip"
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
            id="create-budget-policy-duration"
            componentId="mlflow.gateway.create-budget-policy-modal.duration"
            value={formData.duration}
            onChange={({ target }) => handleFieldChange('duration', target.value as DurationPreset)}
          >
            <SimpleSelectOption value="DAILY">Daily</SimpleSelectOption>
            <SimpleSelectOption value="WEEKLY">Weekly</SimpleSelectOption>
            <SimpleSelectOption value="MONTHLY">Monthly</SimpleSelectOption>
          </SimpleSelect>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="On exceeded" description="Budget on exceeded label" />
          </Typography.Text>
          <SimpleSelect
            id="create-budget-policy-on-exceeded"
            componentId="mlflow.gateway.create-budget-policy-modal.on-exceeded"
            value={formData.budgetAction}
            onChange={({ target }) => handleFieldChange('budgetAction', target.value as BudgetAction)}
          >
            <SimpleSelectOption value="ALERT">Alert</SimpleSelectOption>
            <SimpleSelectOption value="REJECT">Reject requests</SimpleSelectOption>
          </SimpleSelect>
          {formData.budgetAction === 'ALERT' && (
            <Alert
              componentId="mlflow.gateway.create-budget-policy-modal.alert-webhook-info"
              type="info"
              closable={false}
              message={intl.formatMessage({
                defaultMessage:
                  "Please ensure you have a webhook configured to receive budget alert events. You can create one in the 'Budget alert webhooks' section on this page.",
                description: 'Info message when alert action is selected for budget policy',
              })}
            />
          )}
        </div>
      </div>
    </Modal>
  );
};

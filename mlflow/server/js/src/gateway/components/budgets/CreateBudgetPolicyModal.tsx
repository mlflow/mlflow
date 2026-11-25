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
import type { BudgetAction } from '../../types';
import {
  BudgetDurationField,
  DEFAULT_BUDGET_DURATION,
  parseBudgetDuration,
  type BudgetDurationFormValue,
} from './BudgetDurationField';
import {
  BudgetScopeField,
  getBudgetScopePayload,
  useBudgetScopeField,
  type BudgetScopeFormValue,
} from './BudgetScopeField';

interface CreateBudgetPolicyModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

interface FormData extends BudgetScopeFormValue, BudgetDurationFormValue {
  budgetAmount: string;
  budgetAction: BudgetAction;
}

const INITIAL_FORM_DATA: FormData = {
  budgetAmount: '',
  ...DEFAULT_BUDGET_DURATION,
  budgetAction: 'REJECT',
  scope: 'ALL',
  endpointId: '',
  username: '',
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
  const scopeField = useBudgetScopeField(formData);

  const handleClose = useCallback(() => {
    setFormData(INITIAL_FORM_DATA);
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const handleFormChange = useCallback(
    (change: Partial<FormData>) => {
      setFormData((prev) => ({ ...prev, ...change }));
      resetMutation();
    },
    [resetMutation],
  );

  const { durationValue, durationUnit } = formData;
  const duration = useMemo(() => parseBudgetDuration({ durationValue, durationUnit }), [durationValue, durationUnit]);

  const isFormValid = useMemo(() => {
    const amount = parseFloat(formData.budgetAmount);
    return !isNaN(amount) && amount > 0 && scopeField.isValid && duration !== null;
  }, [formData.budgetAmount, scopeField.isValid, duration]);

  const handleSubmit = useCallback(async () => {
    if (!isFormValid || !duration) return;

    try {
      await createBudgetPolicy({
        budget_unit: 'USD',
        budget_amount: parseFloat(formData.budgetAmount),
        duration,
        ...getBudgetScopePayload(formData),
        budget_action: formData.budgetAction,
      });
    } catch {
      // `mutationError` is populated by `useCreateBudgetPolicy` and rendered
      // inline via `errorMessage`. Swallow here so the rejection doesn't bubble
      // to the dev-server overlay / global `unhandledrejection` handler.
      return;
    }
    handleClose();
    onSuccess?.();
  }, [isFormValid, duration, formData, createBudgetPolicy, handleClose, onSuccess]);

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
              onChange={(e) => handleFormChange({ budgetAmount: e.target.value })}
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

        <BudgetScopeField
          componentId="mlflow.gateway.create-budget-policy-modal"
          value={formData}
          onChange={handleFormChange}
          field={scopeField}
        />

        <BudgetDurationField
          componentId="mlflow.gateway.create-budget-policy-modal"
          value={formData}
          onChange={handleFormChange}
        />

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="On exceeded" description="Budget on exceeded label" />
          </Typography.Text>
          <SimpleSelect
            id="create-budget-policy-on-exceeded"
            componentId="mlflow.gateway.create-budget-policy-modal.on-exceeded"
            label={intl.formatMessage({ defaultMessage: 'On exceeded', description: 'Budget on exceeded label' })}
            value={formData.budgetAction}
            onChange={({ target }) => handleFormChange({ budgetAction: target.value as BudgetAction })}
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

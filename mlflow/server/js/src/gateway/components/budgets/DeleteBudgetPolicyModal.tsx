import { useIntl } from 'react-intl';
import { DeleteConfirmationModal } from '../common';
import { useDeleteBudgetPolicy } from '../../hooks/useDeleteBudgetPolicy';
import { formatBudgetPolicySummary } from './budgetFormatUtils';
import type { BudgetPolicy } from '../../types';

interface DeleteBudgetPolicyModalProps {
  open: boolean;
  policy: BudgetPolicy | null;
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteBudgetPolicyModal = ({ open, policy, onClose, onSuccess }: DeleteBudgetPolicyModalProps) => {
  const intl = useIntl();
  const { mutateAsync: deleteBudgetPolicy } = useDeleteBudgetPolicy();

  const handleConfirm = async () => {
    if (!policy) return;
    await deleteBudgetPolicy(policy.budget_policy_id);
    onSuccess?.();
  };

  if (!policy) return null;

  return (
    <DeleteConfirmationModal
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title={intl.formatMessage({
        defaultMessage: 'Delete Budget Policy',
        description: 'Title of the delete budget policy confirmation dialog',
      })}
      itemName={formatBudgetPolicySummary(intl, policy)}
      itemType={intl.formatMessage({
        defaultMessage: 'budget policy',
        description: 'Item type in the delete budget policy confirmation, for example "Delete the budget policy ..."',
      })}
      componentId="mlflow.gateway.delete-budget-policy-modal"
    />
  );
};

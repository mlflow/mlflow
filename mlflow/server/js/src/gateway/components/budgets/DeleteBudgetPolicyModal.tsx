import { DeleteConfirmationModal } from '../common';
import { useDeleteBudgetPolicy } from '../../hooks/useDeleteBudgetPolicy';
import type { BudgetPolicy } from '../../types';

interface DeleteBudgetPolicyModalProps {
  open: boolean;
  policy: BudgetPolicy | null;
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteBudgetPolicyModal = ({ open, policy, onClose, onSuccess }: DeleteBudgetPolicyModalProps) => {
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
      title="Delete Budget Policy"
      itemName={policy.name}
      itemType="budget policy"
      componentIdPrefix="mlflow.gateway.delete-budget-policy-modal"
    />
  );
};

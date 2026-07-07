import { useState, useCallback } from 'react';
import { useBudgetPoliciesQuery } from './useBudgetPoliciesQuery';
import type { BudgetPolicy } from '../types';

export function useBudgetsPage() {
  const { refetch: refetchBudgetPolicies } = useBudgetPoliciesQuery();

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<BudgetPolicy | null>(null);
  const [deletingPolicy, setDeletingPolicy] = useState<BudgetPolicy | null>(null);

  const handleCreateClick = useCallback(() => {
    setIsCreateModalOpen(true);
  }, []);

  const handleCreateModalClose = useCallback(() => {
    setIsCreateModalOpen(false);
  }, []);

  const handleCreateSuccess = useCallback(() => {
    refetchBudgetPolicies();
  }, [refetchBudgetPolicies]);

  const handleEditClick = useCallback((policy: BudgetPolicy) => {
    setEditingPolicy(policy);
  }, []);

  const handleEditModalClose = useCallback(() => {
    setEditingPolicy(null);
  }, []);

  const handleEditSuccess = useCallback(() => {
    refetchBudgetPolicies();
  }, [refetchBudgetPolicies]);

  const handleDeleteClick = useCallback((policy: BudgetPolicy) => {
    setDeletingPolicy(policy);
  }, []);

  const handleDeleteModalClose = useCallback(() => {
    setDeletingPolicy(null);
  }, []);

  const handleDeleteSuccess = useCallback(async () => {
    setDeletingPolicy(null);
    await refetchBudgetPolicies();
  }, [refetchBudgetPolicies]);

  return {
    isCreateModalOpen,
    editingPolicy,
    deletingPolicy,
    handleCreateClick,
    handleCreateModalClose,
    handleCreateSuccess,
    handleEditClick,
    handleEditModalClose,
    handleEditSuccess,
    handleDeleteClick,
    handleDeleteModalClose,
    handleDeleteSuccess,
  };
}

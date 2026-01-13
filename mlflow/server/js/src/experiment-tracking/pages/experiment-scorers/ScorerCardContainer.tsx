import React, { useState, useCallback, useEffect } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import type { ScheduledScorer } from './types';
import { useDeleteScheduledScorerMutation } from './hooks/useDeleteScheduledScorer';
import { syncFormWithScorer, getFormValuesFromScorer } from './scorerCardUtils';
import type { LLMScorerFormData } from './LLMScorerFormRenderer';
import type { CustomCodeScorerFormData } from './CustomCodeScorerFormRenderer';
import { DeleteScorerModalRenderer } from './DeleteScorerModalRenderer';
import ScorerCardRenderer from './ScorerCardRenderer';
import ScorerModalRenderer from './ScorerModalRenderer';
import { SCORER_FORM_MODE } from './constants';

interface ScorerCardContainerProps {
  scorer: ScheduledScorer;
  experimentId: string;
}

const ScorerCardContainer: React.FC<ScorerCardContainerProps> = ({ scorer, experimentId }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeModal, setActiveModal] = useState<'delete' | 'edit' | null>(null);

  // Hook for deleting scorer
  const deleteScorerMutation = useDeleteScheduledScorerMutation();

  // React Hook Form for display mode
  const form = useForm<LLMScorerFormData | CustomCodeScorerFormData>({
    defaultValues: getFormValuesFromScorer(scorer),
  });

  const { control, reset, setValue, getValues } = form;

  // Sync form state with scorer prop changes
  useEffect(() => {
    syncFormWithScorer(scorer, reset);
  }, [scorer, reset]);

  const handleCardClick = useCallback(() => {
    setIsExpanded(!isExpanded);
  }, [isExpanded]);

  const handleExpandToggle = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setIsExpanded(!isExpanded);
    },
    [isExpanded],
  );

  const handleEditClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setActiveModal('edit');
  }, []);

  const handleCloseEditModal = useCallback(() => {
    setActiveModal(null);
  }, []);

  const handleDeleteClick = useCallback(() => {
    setActiveModal('delete');
  }, []);

  const handleDeleteConfirm = useCallback(() => {
    deleteScorerMutation.mutate(
      {
        experimentId,
        scorerNames: [scorer.name],
      },
      {
        onSuccess: () => {
          setActiveModal(null);
        },
      },
    );
  }, [deleteScorerMutation, experimentId, scorer.name]);

  const handleDeleteCancel = useCallback(() => {
    setActiveModal(null);
    deleteScorerMutation.reset();
  }, [deleteScorerMutation]);

  return (
    <FormProvider {...form}>
      <ScorerCardRenderer
        scorer={scorer}
        isExpanded={isExpanded}
        onCardClick={handleCardClick}
        onExpandToggle={handleExpandToggle}
        onEditClick={handleEditClick}
        onDeleteClick={handleDeleteClick}
        control={control}
        setValue={setValue}
        getValues={getValues}
      />
      <ScorerModalRenderer
        visible={activeModal === 'edit'}
        onClose={handleCloseEditModal}
        experimentId={experimentId}
        mode={SCORER_FORM_MODE.EDIT}
        existingScorer={scorer}
      />
      <DeleteScorerModalRenderer
        isOpen={activeModal === 'delete'}
        onClose={handleDeleteCancel}
        onConfirm={handleDeleteConfirm}
        scorer={scorer}
        isLoading={deleteScorerMutation.isLoading}
        error={deleteScorerMutation.error}
      />
    </FormProvider>
  );
};

export default ScorerCardContainer;

/**
 * Scoped "Model settings" sub-view, opened from the composer's model pill.
 *
 * Reuses the setup wizard's provider -> connection steps so the user can switch between any
 * provider (Claude Code, Codex, Ollama, MLflow Gateway) and configure its model/endpoint, but
 * skips the project step (the experiment mapping is unrelated to changing the model and is
 * already configured). On finish/cancel it refetches config so the composer's model pill
 * reflects the change, then returns to chat.
 */

import { useCallback } from 'react';

import { AssistantSetupWizard } from './setup';
import { useAssistantConfigQuery } from './hooks/useAssistantConfigQuery';

interface AssistantModelSettingsProps {
  experimentId?: string;
  onClose: () => void;
}

export const AssistantModelSettings = ({ experimentId, onClose }: AssistantModelSettingsProps) => {
  const { refetch } = useAssistantConfigQuery();

  const handleClose = useCallback(async () => {
    await refetch();
    onClose();
  }, [refetch, onClose]);

  return (
    <AssistantSetupWizard
      experimentId={experimentId}
      initialStep="provider"
      onComplete={handleClose}
      onBack={handleClose}
      skipProject
    />
  );
};

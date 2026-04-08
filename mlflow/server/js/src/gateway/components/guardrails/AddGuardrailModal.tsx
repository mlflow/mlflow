import type { GatewayGuardrailConfig } from '../../types';

interface GuardrailModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onDelete?: (guardrailId: string) => void;
  endpointName: string;
  endpointId: string;
  editingGuardrail?: GatewayGuardrailConfig | null;
  experimentId?: string;
}

/**
 * Placeholder for the guardrail create/edit modal.
 * The full implementation is added in a follow-up PR.
 */
export const AddGuardrailModal = (_props: GuardrailModalProps) => {
  return null;
};

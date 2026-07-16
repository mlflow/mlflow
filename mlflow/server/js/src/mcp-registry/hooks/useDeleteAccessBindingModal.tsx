import { useState } from 'react';
import { useIntl } from 'react-intl';
import { FormattedMessage } from 'react-intl';
import type { MCPAccessBinding } from '../types';
import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { useDeleteAccessBindingMutation } from './useAccessBindingMutation';

export const useDeleteAccessBindingModal = ({ serverName }: { serverName: string }) => {
  const intl = useIntl();
  const [binding, setBinding] = useState<MCPAccessBinding | undefined>(undefined);
  const mutation = useDeleteAccessBindingMutation();

  const DeleteAccessBindingModal = (
    <ConfirmationModal
      componentId="mlflow.mcp_registry.detail.delete_binding_modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete access endpoint',
        description: 'Access endpoint delete confirmation modal title',
      })}
      visible={Boolean(binding)}
      message={
        <FormattedMessage
          defaultMessage="Are you sure you want to delete this access endpoint? This action cannot be undone."
          description="Access endpoint delete confirmation message"
        />
      }
      isLoading={mutation.isLoading}
      error={mutation.error?.message ?? null}
      onConfirm={() => {
        if (binding) {
          mutation.mutate({ serverName, bindingId: binding.binding_id }, { onSuccess: () => setBinding(undefined) });
        }
      }}
      onCancel={() => {
        mutation.reset();
        setBinding(undefined);
      }}
    />
  );

  return { DeleteAccessBindingModal, openDeleteBinding: (b: MCPAccessBinding) => setBinding(b) };
};

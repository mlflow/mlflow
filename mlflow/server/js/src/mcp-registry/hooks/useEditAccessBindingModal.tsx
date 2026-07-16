import { useState } from 'react';
import type { MCPAccessBinding } from '../types';
import { AccessBindingModal } from '../components/AccessBindingModal';

export const useEditAccessBindingModal = ({ serverName }: { serverName: string }) => {
  const [binding, setBinding] = useState<MCPAccessBinding | undefined>(undefined);

  const EditAccessBindingModal = (
    <AccessBindingModal
      visible={Boolean(binding)}
      onCancel={() => setBinding(undefined)}
      editBinding={binding}
      lockedServer={serverName}
    />
  );

  return { EditAccessBindingModal, openEditBinding: (b: MCPAccessBinding) => setBinding(b) };
};

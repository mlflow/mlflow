import { useState } from 'react';
import type { MCPAccessBinding } from '../types';
import { AccessBindingModal } from '../components/AccessBindingModal';

export const useEditAccessBindingModal = ({
  serverName,
  scopedVersion,
  scopedAliases,
}: {
  serverName: string;
  scopedVersion?: string;
  scopedAliases?: string[];
}) => {
  const [binding, setBinding] = useState<MCPAccessBinding | undefined>(undefined);

  const EditAccessBindingModal = (
    <AccessBindingModal
      visible={Boolean(binding)}
      onCancel={() => setBinding(undefined)}
      editBinding={binding}
      lockedServer={serverName}
      scopedVersion={scopedVersion}
      scopedAliases={scopedAliases}
    />
  );

  return { EditAccessBindingModal, openEditBinding: (b: MCPAccessBinding) => setBinding(b) };
};

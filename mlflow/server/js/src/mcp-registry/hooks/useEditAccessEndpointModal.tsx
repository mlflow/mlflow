import { useState } from 'react';
import type { MCPAccessEndpoint } from '../types';
import { AccessEndpointModal } from '../components/AccessEndpointModal';

export const useEditAccessEndpointModal = ({
  serverName,
  scopedVersion,
  scopedAliases,
}: {
  serverName: string;
  scopedVersion?: string;
  scopedAliases?: string[];
}) => {
  const [endpoint, setEndpoint] = useState<MCPAccessEndpoint | undefined>(undefined);

  const EditAccessEndpointModal = (
    <AccessEndpointModal
      visible={Boolean(endpoint)}
      onCancel={() => setEndpoint(undefined)}
      editEndpoint={endpoint}
      lockedServer={serverName}
      scopedVersion={scopedVersion}
      scopedAliases={scopedAliases}
    />
  );

  return { EditAccessEndpointModal, openEditEndpoint: (b: MCPAccessEndpoint) => setEndpoint(b) };
};

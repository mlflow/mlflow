import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { AccessEndpointModal } from '../components/AccessEndpointModal';

export const useAddAccessEndpointModal = ({
  serverName,
  scopedVersion,
  scopedAliases,
}: {
  serverName: string;
  scopedVersion?: string;
  scopedAliases?: string[];
}) => {
  const [open, setOpen] = useState(false);

  const AddAccessEndpointModal = (
    <AccessEndpointModal
      visible={open}
      onCancel={() => setOpen(false)}
      lockedServer={serverName}
      scopedVersion={scopedVersion}
      scopedAliases={scopedAliases}
      createTitle={
        <FormattedMessage
          defaultMessage="Add access endpoint"
          description="MCP server add access endpoint modal title"
        />
      }
    />
  );

  return { AddAccessEndpointModal, openAddEndpoint: () => setOpen(true) };
};

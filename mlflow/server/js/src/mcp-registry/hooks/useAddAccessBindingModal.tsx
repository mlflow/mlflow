import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { AccessBindingModal } from '../components/AccessBindingModal';

export const useAddAccessBindingModal = ({
  serverName,
  scopedVersion,
  scopedAliases,
}: {
  serverName: string;
  scopedVersion?: string;
  scopedAliases?: string[];
}) => {
  const [open, setOpen] = useState(false);

  const AddAccessBindingModal = (
    <AccessBindingModal
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

  return { AddAccessBindingModal, openAddBinding: () => setOpen(true) };
};

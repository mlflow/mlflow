import { useState } from 'react';
import { UpdateVersionDisplayNameModal } from '../components/UpdateVersionDisplayNameModal';
import { useUpdateMCPServerDisplayName } from './useMCPServerVersionMutations';

export const useEditDisplayNameModal = ({ serverName }: { serverName: string }) => {
  const [currentDisplayName, setCurrentDisplayName] = useState('');
  const [visible, setVisible] = useState(false);
  const mutation = useUpdateMCPServerDisplayName(serverName);

  const openEditDisplayName = (displayName: string) => {
    setCurrentDisplayName(displayName);
    setVisible(true);
  };

  const EditDisplayNameModal = (
    <UpdateVersionDisplayNameModal
      visible={visible}
      currentDisplayName={currentDisplayName}
      isLoading={mutation.isLoading}
      error={mutation.error}
      onUpdate={(newDisplayName) => {
        mutation.mutate(newDisplayName || null, {
          onSuccess: () => setVisible(false),
        });
      }}
      onCancel={() => {
        mutation.reset();
        setVisible(false);
      }}
    />
  );

  return { EditDisplayNameModal, openEditDisplayName };
};

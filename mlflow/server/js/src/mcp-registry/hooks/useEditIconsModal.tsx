import { useRef, useState } from 'react';
import { Alert, Modal, Spacer } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import type { MCPIcon } from '../types';
import { MCPRegistryApi } from '../api';
import { MCP_QUERY_KEYS } from '../utils';
import { IconEditor } from '../components/IconEditor';

export const useEditIconsModal = ({ serverName }: { serverName: string }) => {
  const queryClient = useQueryClient();
  const initialRef = useRef<MCPIcon[]>([]);
  const serverJsonIconsRef = useRef<MCPIcon[] | undefined>(undefined);
  const [visible, setVisible] = useState(false);
  const [icons, setIcons] = useState<MCPIcon[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const openEditIcons = (currentIcons: MCPIcon[], serverJsonIcons?: MCPIcon[]) => {
    initialRef.current = currentIcons;
    serverJsonIconsRef.current = serverJsonIcons;
    setIcons(currentIcons);
    setError(null);
    setVisible(true);
  };

  const handleSave = async () => {
    const validIcons = icons.filter((i) => i.src.trim());
    setIsSaving(true);
    setError(null);
    try {
      await MCPRegistryApi.updateMCPServer(serverName, {
        icons: validIcons.length > 0 ? validIcons : null,
      });
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVER, serverName]);
      queryClient.invalidateQueries([MCP_QUERY_KEYS.SERVERS_LIST]);
      initialRef.current = validIcons;
      setVisible(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setIcons(initialRef.current);
    setError(null);
    setVisible(false);
  };

  const EditIconsModal = visible ? (
    <Modal
      componentId="mlflow.mcp_registry.edit_icons_modal"
      title={<FormattedMessage defaultMessage="Edit icon" description="Edit icon modal title" />}
      visible={visible}
      size="wide"
      destroyOnClose
      confirmLoading={isSaving}
      okText={<FormattedMessage defaultMessage="Save" description="Edit icons modal save button" />}
      onOk={handleSave}
      onCancel={handleCancel}
    >
      {error && (
        <>
          <Alert
            componentId="mlflow.mcp_registry.edit_icons_modal.error"
            type="error"
            closable
            onClose={() => setError(null)}
            message={error}
          />
          <Spacer size="sm" />
        </>
      )}
      <IconEditor icons={icons} onChange={setIcons} serverJsonIcons={serverJsonIconsRef.current} />
    </Modal>
  ) : null;

  return { EditIconsModal, openEditIcons };
};

import { useEffect, useRef, useState } from 'react';
import type { ConnectOptionKey, ConnectOptionsMap, MCPServerVersion } from '../types';
import { useUpdateMCPServerVersion } from './useMCPServerVersionMutations';

export const useConnectOptionToggle = (serverName: string, version?: MCPServerVersion) => {
  const [localConnectOptions, setLocalConnectOptions] = useState<ConnectOptionsMap | undefined>(undefined);
  const currentVersionRef = useRef(version?.version);
  currentVersionRef.current = version?.version;

  useEffect(() => {
    setLocalConnectOptions(undefined);
  }, [version?.version]);

  const updateVersionMutation = useUpdateMCPServerVersion(serverName);

  const connectOptions = localConnectOptions ?? version?.connect_options ?? undefined;

  const handleToggleConnectOption = (key: ConnectOptionKey, visible: boolean) => {
    if (!version) return;
    const toggledVersion = version.version;
    const current = localConnectOptions ?? version.connect_options ?? {};
    const updated: ConnectOptionsMap = { ...current, [key]: { hidden: !visible } };
    setLocalConnectOptions(updated);
    updateVersionMutation.mutate(
      { version: toggledVersion, connectOptions: updated },
      {
        onError: () => {
          if (currentVersionRef.current === toggledVersion) {
            setLocalConnectOptions(current);
          }
        },
      },
    );
  };

  return { connectOptions, handleToggleConnectOption };
};

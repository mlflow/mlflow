import { useMemo } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

import type { MCPTool, ServerJSONPayload } from '../types';
import { deriveClientName } from '../installInstructions';
import { RemotesSubsection } from './RemotesSubsection';
import { PackagesSubsection } from './PackagesSubsection';
import { ToolsSubsection } from './ToolsSubsection';
import { RawJSONToggle, RawToolsJSONToggle } from './JSONToggles';

export const ServerJSONSection = ({
  serverJson,
  serverName,
  isAdmin,
  isAuthAvailable,
  hiddenConnectOptions,
  onToggleConnectOption,
}: {
  serverJson: ServerJSONPayload;
  serverName: string;
  isAdmin?: boolean;
  isAuthAvailable?: boolean;
  hiddenConnectOptions?: string[];
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const packages = serverJson.packages ?? [];
  const remotes = serverJson.remotes ?? [];
  const derivedName = useMemo(() => deriveClientName(serverName), [serverName]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {remotes.length > 0 && (
        <RemotesSubsection
          remotes={remotes}
          serverName={serverName}
          derivedName={derivedName}
          serverJson={serverJson}
          isAdmin={isAdmin}
          isAuthAvailable={isAuthAvailable}
          hiddenConnectOptions={hiddenConnectOptions}
          onToggleConnectOption={onToggleConnectOption}
        />
      )}
      {packages.length > 0 && (
        <PackagesSubsection
          packages={packages}
          serverName={serverName}
          derivedName={derivedName}
          serverJson={serverJson}
          isAdmin={isAdmin}
          isAuthAvailable={isAuthAvailable}
          hiddenConnectOptions={hiddenConnectOptions}
          onToggleConnectOption={onToggleConnectOption}
        />
      )}
      {(!isAuthAvailable || isAdmin) && <RawJSONToggle serverJson={serverJson} />}
    </div>
  );
};

export const ToolsSection = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <ToolsSubsection tools={tools} />
      <RawToolsJSONToggle tools={tools} />
    </div>
  );
};

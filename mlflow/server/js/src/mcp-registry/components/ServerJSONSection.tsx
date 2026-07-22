import { useMemo } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { flexColumnGapStyles } from '../styles';

import type { MCPServer, MCPServerVersion, MCPTool, ServerJSONPayload } from '../types';
import { deriveClientName } from '../installInstructions';
import { useServerState } from '../hooks/useServerState';
import { useConnectOptionToggle } from '../hooks/useConnectOptionToggle';
import { RemotesSubsection } from './RemotesSubsection';
import { PackagesSubsection } from './PackagesSubsection';
import { ToolsSubsection } from './ToolsSubsection';
import { RawJSONToggle, RawToolsJSONToggle } from './JSONToggles';

export const ServerJSONSection = ({
  serverJson,
  server,
  version,
}: {
  serverJson: ServerJSONPayload;
  server: MCPServer;
  version?: MCPServerVersion;
}) => {
  const { theme } = useDesignSystemTheme();
  const { showVisibilityControls, canUpdate } = useServerState(server);
  const { connectOptions, handleToggleConnectOption } = useConnectOptionToggle(server.name, version);
  const packages = serverJson.packages ?? [];
  const remotes = serverJson.remotes ?? [];
  const derivedName = useMemo(() => deriveClientName(server.name), [server.name]);

  return (
    <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
      {remotes.length > 0 && (
        <RemotesSubsection
          remotes={remotes}
          derivedName={derivedName}
          showVisibilityControls={showVisibilityControls}
          connectOptions={connectOptions}
          onToggleConnectOption={handleToggleConnectOption}
        />
      )}
      {packages.length > 0 && (
        <PackagesSubsection
          packages={packages}
          derivedName={derivedName}
          showVisibilityControls={showVisibilityControls}
          connectOptions={connectOptions}
          onToggleConnectOption={handleToggleConnectOption}
        />
      )}
      {canUpdate && <RawJSONToggle serverJson={serverJson} />}
    </div>
  );
};

export const ToolsSection = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
      <ToolsSubsection tools={tools} />
      <RawToolsJSONToggle tools={tools} />
    </div>
  );
};

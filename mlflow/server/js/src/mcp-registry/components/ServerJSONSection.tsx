import { useMemo } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { flexColumnGapStyles } from '../styles';

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
  connectOptions,
  onToggleConnectOption,
}: {
  serverJson: ServerJSONPayload;
  serverName: string;
  isAdmin?: boolean;
  isAuthAvailable?: boolean;
  connectOptions?: Record<string, { hidden?: boolean }>;
  onToggleConnectOption?: (key: string, visible: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const packages = serverJson.packages ?? [];
  const remotes = serverJson.remotes ?? [];
  const derivedName = useMemo(() => deriveClientName(serverName), [serverName]);

  return (
    <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
      {remotes.length > 0 && (
        <RemotesSubsection
          remotes={remotes}
          derivedName={derivedName}
          isAdmin={isAdmin}
          isAuthAvailable={isAuthAvailable}
          connectOptions={connectOptions}
          onToggleConnectOption={onToggleConnectOption}
        />
      )}
      {packages.length > 0 && (
        <PackagesSubsection
          packages={packages}
          derivedName={derivedName}
          isAdmin={isAdmin}
          isAuthAvailable={isAuthAvailable}
          connectOptions={connectOptions}
          onToggleConnectOption={onToggleConnectOption}
        />
      )}
      <RawJSONToggle serverJson={serverJson} />
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

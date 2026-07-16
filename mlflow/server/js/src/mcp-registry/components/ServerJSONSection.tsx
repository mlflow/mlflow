import { useMemo } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { flexColumnGapStyles } from '../styles';

import type { MCPAccessBinding, MCPTool, ServerJSONPayload } from '../types';
import { deriveClientName } from '../installInstructions';
import { AccessBindingsSubsection } from './AccessBindingsSubsection';
import { RemotesSubsection } from './RemotesSubsection';
import { PackagesSubsection } from './PackagesSubsection';
import { ToolsSubsection } from './ToolsSubsection';
import { RawJSONToggle, RawToolsJSONToggle } from './JSONToggles';

export const ServerJSONSection = ({
  serverJson,
  serverName,
  bindings,
  isAdmin,
  isAuthAvailable,
  connectOptions,
  onToggleConnectOption,
  onAddBinding,
  onEditBinding,
  onDeleteBinding,
}: {
  serverJson: ServerJSONPayload;
  serverName: string;
  bindings?: MCPAccessBinding[];
  isAdmin?: boolean;
  isAuthAvailable?: boolean;
  connectOptions?: Record<string, { hidden?: boolean }>;
  onToggleConnectOption?: (key: string, visible: boolean) => void;
  onAddBinding?: () => void;
  onEditBinding?: (binding: MCPAccessBinding) => void;
  onDeleteBinding?: (binding: MCPAccessBinding) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const packages = serverJson.packages ?? [];
  const remotes = serverJson.remotes ?? [];
  const derivedName = useMemo(() => deriveClientName(serverName), [serverName]);
  const showVisibilityControls = isAuthAvailable && isAdmin;

  return (
    <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
      <AccessBindingsSubsection
        bindings={bindings ?? []}
        derivedName={derivedName}
        isAdmin={isAdmin}
        isAuthAvailable={isAuthAvailable}
        onAddBinding={onAddBinding}
        onEditBinding={onEditBinding}
        onDeleteBinding={onDeleteBinding}
      />
      {remotes.length > 0 && (
        <RemotesSubsection
          remotes={remotes}
          derivedName={derivedName}
          showVisibilityControls={showVisibilityControls}
          connectOptions={connectOptions}
          onToggleConnectOption={onToggleConnectOption}
        />
      )}
      {packages.length > 0 && (
        <PackagesSubsection
          packages={packages}
          derivedName={derivedName}
          showVisibilityControls={showVisibilityControls}
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

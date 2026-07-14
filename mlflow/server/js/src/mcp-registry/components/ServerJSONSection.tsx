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
}: {
  serverJson: ServerJSONPayload;
  serverName: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const packages = serverJson.packages ?? [];
  const remotes = serverJson.remotes ?? [];
  const derivedName = useMemo(() => deriveClientName(serverName), [serverName]);

  return (
    <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
      {remotes.length > 0 && <RemotesSubsection remotes={remotes} derivedName={derivedName} />}
      {packages.length > 0 && <PackagesSubsection packages={packages} derivedName={derivedName} />}
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

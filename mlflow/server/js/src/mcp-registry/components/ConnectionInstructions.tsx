import { useMemo, useState } from 'react';
import {
  CopyIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { ConnectionFormat, ConnectionSource } from '../types';
import type { ServerJSONPackage, ServerJSONPayload, ServerJSONTransport } from '../types';
import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { buildPackageInstruction, buildRemoteInstruction, formatMcpJsonSnippet } from '../installInstructions';
import type { InstructionBlock } from '../installInstructions';

export type ConnectionInstructionsProps = {
  derivedName: string;
  detailLink?: React.ReactNode;
} & (
  | { source: ConnectionSource.PACKAGE; pkg: NonNullable<ServerJSONPayload['packages']>[number] }
  | { source: ConnectionSource.REMOTE; remote: NonNullable<ServerJSONPayload['remotes']>[number] }
);

export const ConnectionInstructions = (props: ConnectionInstructionsProps) => {
  const { theme } = useDesignSystemTheme();

  const { source, derivedName } = props;
  const sourceData = source === ConnectionSource.PACKAGE ? props.pkg : props.remote;
  const block = useMemo((): InstructionBlock => {
    switch (source) {
      case ConnectionSource.PACKAGE:
        return buildPackageInstruction(sourceData as ServerJSONPackage, derivedName);
      case ConnectionSource.REMOTE:
        return buildRemoteInstruction(sourceData as ServerJSONTransport, derivedName);
    }
  }, [source, sourceData, derivedName]);

  const [format, setFormat] = useState<ConnectionFormat>(
    block.claudeCodeCommand ? ConnectionFormat.CLAUDE_CODE : ConnectionFormat.MCP_JSON,
  );

  const snippet = format === ConnectionFormat.CLAUDE_CODE
    ? (block.claudeCodeCommand ?? '')
    : (block.mcpJsonConfig ? formatMcpJsonSnippet(derivedName, block.mcpJsonConfig) : '');

  const hasBothFormats = block.claudeCodeCommand != null && block.mcpJsonConfig != null;
  const hasAnySnippet = block.claudeCodeCommand != null || block.mcpJsonConfig != null;

  if (!hasAnySnippet) {
    return block.fallbackReason ? (
      <div css={{ padding: theme.spacing.sm, backgroundColor: theme.colors.backgroundSecondary, borderRadius: theme.borders.borderRadiusSm }}>
        <Typography.Text color="secondary" size="sm">{block.fallbackReason}</Typography.Text>
        {block.fallbackUrl && (
          <div css={{ marginTop: theme.spacing.xs }}>
            <a href={block.fallbackUrl} target="_blank" rel="noopener noreferrer">
              <FormattedMessage defaultMessage="View documentation" description="Link to documentation for unsupported install method" />
            </a>
          </div>
        )}
      </div>
    ) : null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {(hasBothFormats || props.detailLink) && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          {hasBothFormats && (
            <SegmentedControlGroup
              name="mlflow.mcp_registry.detail.connection_format"
              componentId="mlflow.mcp_registry.detail.connection_format"
              value={format}
              onChange={(e) => setFormat(e.target.value as ConnectionFormat)}
            >
              <SegmentedControlButton value={ConnectionFormat.CLAUDE_CODE}>
                Claude Code
              </SegmentedControlButton>
              <SegmentedControlButton value={ConnectionFormat.MCP_JSON}>
                .mcp.json
              </SegmentedControlButton>
            </SegmentedControlGroup>
          )}
          {props.detailLink}
        </div>
      )}
      {snippet && (
        <div css={{ position: 'relative' }}>
          <CopyButton
            componentId="mlflow.mcp_registry.detail.connection_snippet.copy_button"
            showLabel={false}
            copyText={snippet}
            icon={<CopyIcon />}
            css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          />
          <CodeSnippet
            language={format === ConnectionFormat.MCP_JSON ? 'json' : 'text'}
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{ padding: theme.spacing.sm, paddingRight: theme.spacing.xl + theme.spacing.sm }}
          >
            {snippet}
          </CodeSnippet>
        </div>
      )}
      {block.notes.length > 0 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {block.notes.map((note, i) => (
            <Typography.Text key={i} color="secondary" size="sm">{note}</Typography.Text>
          ))}
        </div>
      )}
    </div>
  );
};

import { useMemo, useState } from 'react';
import {
  CopyIcon,
  InfoIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { sanitizeHref } from '../utils';

import { ConnectionFormat, ConnectionSource } from '../types';
import type { MCPAccessEndpoint, ServerJSONPayload, TransportType } from '../types';
import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { buildPackageInstruction, buildRemoteInstruction, formatMcpJsonSnippet } from '../installInstructions';
import {
  fallbackInfoBoxStyles,
  overlayButtonStyles,
  flexColumnGapStyles,
  flexRowStyles,
  mcpIconStyles,
  noShrinkStyles,
} from '../styles';
import type { InstructionBlock } from '../installInstructions';

export type ConnectionInstructionsProps = {
  derivedName: string;
  detailLink?: React.ReactNode;
} & (
  | { source: ConnectionSource.PACKAGE; pkg: NonNullable<ServerJSONPayload['packages']>[number] }
  | { source: ConnectionSource.REMOTE; remote: NonNullable<ServerJSONPayload['remotes']>[number] }
  | { source: ConnectionSource.ENDPOINT; endpoint: MCPAccessEndpoint }
);

export const ConnectionInstructions = (props: ConnectionInstructionsProps) => {
  const { theme } = useDesignSystemTheme();

  const { source, derivedName } = props;
  const pkg = source === ConnectionSource.PACKAGE ? props.pkg : undefined;
  const remote = source === ConnectionSource.REMOTE ? props.remote : undefined;
  const endpoint = source === ConnectionSource.ENDPOINT ? props.endpoint : undefined;

  const block = useMemo((): InstructionBlock => {
    if (source === ConnectionSource.PACKAGE && pkg) {
      return buildPackageInstruction(pkg, derivedName);
    }
    if (source === ConnectionSource.ENDPOINT && endpoint) {
      return buildRemoteInstruction({ type: endpoint.transport_type as TransportType, url: endpoint.url }, derivedName);
    }
    if (remote) {
      return buildRemoteInstruction(remote, derivedName);
    }
    return {
      kind: 'fallback',
      label: '',
      claudeCodeCommand: null,
      mcpJsonConfig: null,
      notes: [],
    };
  }, [source, derivedName, pkg, remote, endpoint]);

  const [format, setFormat] = useState<ConnectionFormat>(
    block.claudeCodeCommand ? ConnectionFormat.CLAUDE_CODE : ConnectionFormat.MCP_JSON,
  );

  const snippet =
    format === ConnectionFormat.CLAUDE_CODE
      ? (block.claudeCodeCommand ?? '')
      : block.mcpJsonConfig
        ? formatMcpJsonSnippet(derivedName, block.mcpJsonConfig)
        : '';

  const hasBothFormats = block.claudeCodeCommand != null && block.mcpJsonConfig != null;
  const hasAnySnippet = block.claudeCodeCommand != null || block.mcpJsonConfig != null;

  if (!hasAnySnippet) {
    return block.fallbackReason ? (
      <div css={fallbackInfoBoxStyles(theme)}>
        <InfoIcon css={{ ...mcpIconStyles(theme), ...noShrinkStyles }} />
        <div>
          <Typography.Text color="secondary" size="sm">
            {block.fallbackReason}
          </Typography.Text>
          {sanitizeHref(block.fallbackUrl) && (
            <div css={{ marginTop: theme.spacing.xs }}>
              <a href={sanitizeHref(block.fallbackUrl)} target="_blank" rel="noopener noreferrer">
                <FormattedMessage
                  defaultMessage="View setup instructions"
                  description="Link to publisher documentation for manual install"
                />
              </a>
            </div>
          )}
        </div>
      </div>
    ) : null;
  }

  return (
    <div css={flexColumnGapStyles(theme)}>
      {(hasBothFormats || props.detailLink) && (
        <div css={flexRowStyles(theme)}>
          {hasBothFormats && (
            <SegmentedControlGroup
              name="mlflow.mcp_registry.detail.connection_format"
              componentId="mlflow.mcp_registry.detail.connection_format"
              value={format}
              onChange={(e) => setFormat(e.target.value as ConnectionFormat)}
            >
              <SegmentedControlButton value={ConnectionFormat.CLAUDE_CODE}>
                <FormattedMessage defaultMessage="Claude Code" description="Claude Code install format label" />
              </SegmentedControlButton>
              <SegmentedControlButton value={ConnectionFormat.MCP_JSON}>
                <FormattedMessage defaultMessage=".mcp.json" description="mcp.json install format label" />
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
            css={overlayButtonStyles(theme)}
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
        <div css={flexColumnGapStyles(theme, theme.spacing.xs)}>
          {block.notes.map((note, i) => (
            <Typography.Text key={i} color="secondary" size="sm">
              {note}
            </Typography.Text>
          ))}
        </div>
      )}
    </div>
  );
};

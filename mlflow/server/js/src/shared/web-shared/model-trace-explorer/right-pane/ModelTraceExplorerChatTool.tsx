import { useState } from 'react';

import { ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerChatToolParam } from './ModelTraceExplorerChatToolParam';
import type { ModelTraceChatTool } from '../ModelTrace.types';

export function ModelTraceExplorerChatTool({ tool }: { tool: ModelTraceChatTool }) {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);

  const description = tool.function.description;
  const paramProperties = tool.function.parameters?.properties;
  const requiredParams = tool.function.parameters?.required ?? [];

  // tools only need to have names, so it's
  // possible that no additional info exists
  const isExpandable = description || paramProperties;

  const hoverStyles = isExpandable
    ? { ':hover': { backgroundColor: theme.colors.actionIconBackgroundHover, cursor: 'pointer' } }
    : {};

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
      data-testid="model-trace-explorer-chat-tool"
    >
      <div
        role="button"
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.sm,
          alignItems: 'center',
          borderBottom: isExpandable && expanded ? `1px solid ${theme.colors.border}` : 'none',
          padding: theme.spacing.sm,
          ...hoverStyles,
        }}
        onClick={() => setExpanded(!expanded)}
        data-testid="model-trace-explorer-chat-tool-toggle"
      >
        {isExpandable && (expanded ? <ChevronDownIcon /> : <ChevronRightIcon />)}
        <Typography.Text
          bold
          withoutMargins
          style={{ whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}
        >
          {tool.function.name}
        </Typography.Text>
      </div>
      {isExpandable && expanded && (
        <div css={{ padding: theme.spacing.sm }}>
          {description && (
            <Typography.Paragraph
              style={{ whiteSpace: 'pre-wrap', marginBottom: theme.spacing.sm, padding: `0px ${theme.spacing.xs}px` }}
            >
              {tool.function.description}
            </Typography.Paragraph>
          )}
          {paramProperties && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {Object.keys(paramProperties).map((key) => (
                <ModelTraceExplorerChatToolParam
                  key={key}
                  paramName={key}
                  paramProperties={paramProperties[key]}
                  isRequired={requiredParams.includes(key)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

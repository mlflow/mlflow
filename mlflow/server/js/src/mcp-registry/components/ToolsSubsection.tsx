import { useState } from 'react';
import { Button, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPTool } from '../types';
import { tagListStyles, ellipsisStyles, noShrinkStyles, showMoreRowStyles } from '../styles';
import { InputSchemaToggle, OutputSchemaToggle } from './JSONToggles';
import { ExpandableListSection } from './ExpandableListSection';

const INITIAL_VISIBLE_TOOLS = 10;

export const ToolsSubsection = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [showAll, setShowAll] = useState(false);
  const visibleTools = showAll ? tools : tools.slice(0, INITIAL_VISIBLE_TOOLS);
  const hiddenCount = tools.length - INITIAL_VISIBLE_TOOLS;

  return (
    <ExpandableListSection
      items={visibleTools}
      getKey={(tool) => tool.name}
      getAriaLabel={(tool, expanded) =>
        intl.formatMessage(
          {
            defaultMessage: '{action} tool {name}',
            description: 'Aria label for expanding/collapsing a tool row',
          },
          { action: expanded ? 'Collapse' : 'Expand', name: tool.name },
        )
      }
      renderRow={({ item: tool }) => (
        <>
          <Tag componentId="mlflow.mcp_registry.detail.tool_name_tag" color="turquoise" css={noShrinkStyles}>
            {tool.name}
          </Tag>
          {tool.description && (
            <Typography.Text color="secondary" size="sm" css={ellipsisStyles(theme)}>
              {tool.description}
            </Typography.Text>
          )}
        </>
      )}
      renderExpanded={(tool) => (
        <>
          {tool.annotations && Object.keys(tool.annotations).length > 0 && (
            <div css={tagListStyles(theme)}>
              {Object.entries(tool.annotations).map(([key, value]) => (
                <Tag key={key} componentId="mlflow.mcp_registry.detail.tool_annotation_tag">
                  {key}: {String(value)}
                </Tag>
              ))}
            </div>
          )}
          {tool.inputSchema && Object.keys(tool.inputSchema).length > 0 && (
            <InputSchemaToggle data={tool.inputSchema} />
          )}
          {tool.outputSchema && Object.keys(tool.outputSchema).length > 0 && (
            <OutputSchemaToggle data={tool.outputSchema} />
          )}
        </>
      )}
      footer={
        hiddenCount > 0 ? (
          <div css={showMoreRowStyles(theme)}>
            <Button
              componentId="mlflow.mcp_registry.detail.toggle_tools"
              type="link"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? (
                <FormattedMessage defaultMessage="Show less" description="Show less tools button" />
              ) : (
                <FormattedMessage
                  defaultMessage="Show {count} more"
                  description="Show more tools button"
                  values={{ count: hiddenCount }}
                />
              )}
            </Button>
          </div>
        ) : undefined
      }
    />
  );
};

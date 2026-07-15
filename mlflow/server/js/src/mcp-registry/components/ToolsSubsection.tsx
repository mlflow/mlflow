import { useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPTool } from '../types';
import {
  borderedSectionContainerStyles,
  chevronContainerStyles,
  expandableRowButtonStyles,
  expandedContentPanelStyles,
  showMoreRowStyles,
  ellipsisStyles,
  tagListStyles,
  noShrinkStyles,
} from '../styles';
import { InputSchemaToggle, OutputSchemaToggle } from './JSONToggles';

const INITIAL_VISIBLE_TOOLS = 10;

export const ToolsSubsection = ({ tools }: { tools: MCPTool[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const visibleTools = showAll ? tools : tools.slice(0, INITIAL_VISIBLE_TOOLS);
  const hiddenCount = tools.length - INITIAL_VISIBLE_TOOLS;

  return (
    <div>
      <div css={borderedSectionContainerStyles(theme)}>
        {visibleTools.map((tool, index) => (
          <div
            key={tool.name}
            css={{
              borderTop: index > 0 ? `1px solid ${theme.colors.border}` : 'none',
            }}
          >
            <button
              type="button"
              onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
              aria-expanded={expandedIndex === index}
              aria-label={intl.formatMessage(
                {
                  defaultMessage: '{action} tool {name}',
                  description: 'Aria label for expanding/collapsing a tool row',
                },
                {
                  action: expandedIndex === index ? 'Collapse' : 'Expand',
                  name: tool.name,
                },
              )}
              css={expandableRowButtonStyles(theme)}
            >
              <div css={chevronContainerStyles(theme)}>
                {expandedIndex === index ? <ChevronDownIcon /> : <ChevronRightIcon />}
              </div>
              <Tag componentId="mlflow.mcp_registry.detail.tool_name_tag" color="turquoise" css={noShrinkStyles}>
                {tool.name}
              </Tag>
              {tool.description && (
                <Typography.Text color="secondary" size="sm" css={ellipsisStyles(theme)}>
                  {tool.description}
                </Typography.Text>
              )}
            </button>

            {expandedIndex === index && (
              <div css={expandedContentPanelStyles(theme)}>
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
              </div>
            )}
          </div>
        ))}
        {hiddenCount > 0 && (
          <div css={showMoreRowStyles(theme)}>
            <Button
              componentId="mlflow.mcp_registry.detail.toggle_tools"
              type="link"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? (
                <FormattedMessage
                  defaultMessage="Show less"
                  description="MCP server version detail show less tools button"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Show {count} more"
                  description="MCP server version detail show more tools button"
                  values={{ count: hiddenCount }}
                />
              )}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

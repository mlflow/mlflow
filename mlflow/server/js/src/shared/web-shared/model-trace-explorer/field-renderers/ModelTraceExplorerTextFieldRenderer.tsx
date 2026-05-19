import { useState } from 'react';

import { ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';

const STRING_TRUNCATION_LIMIT = 400;

export const ModelTraceExplorerTextFieldRenderer = ({ title, value }: { title: string; value: string }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);

  const isExpandable = value.length > STRING_TRUNCATION_LIMIT;
  const displayValue =
    !expanded && value.length > STRING_TRUNCATION_LIMIT ? value.slice(0, STRING_TRUNCATION_LIMIT) + '...' : value;

  return (
    <div
      css={{
        borderRadius: theme.borders.borderRadiusSm,
      }}
    >
      {title && (
        <div
          role="button"
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingInline: theme.spacing.sm,
            marginBottom: theme.spacing.xs,
          }}
        >
          <Typography.Title css={{ marginLeft: theme.spacing.xs }} level={4} color="secondary" withoutMargins>
            {title}
          </Typography.Title>
        </div>
      )}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          marginInline: theme.spacing.sm,
          paddingInline: theme.spacing.sm,
          paddingBlock: theme.spacing.sm,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusSm,
          // get rid of last margin in markdown renderer
          '& > div:last-of-type': {
            marginBottom: 0,
          },
        }}
      >
        <GenAIMarkdownRenderer>{displayValue}</GenAIMarkdownRenderer>
        {isExpandable && (
          <Typography.Link
            onClick={() => setExpanded(!expanded)}
            componentId="shared.model-trace-explorer.text-field-see-more-link"
          >
            {expanded ? (
              <FormattedMessage
                defaultMessage="See less"
                description="Button to collapse a long text field in the trace explorer summary field renderer"
              />
            ) : (
              <FormattedMessage
                defaultMessage="See more"
                description="Button to expand a long text field in the trace explorer summary field renderer"
              />
            )}
          </Typography.Link>
        )}
      </div>
    </div>
  );
};

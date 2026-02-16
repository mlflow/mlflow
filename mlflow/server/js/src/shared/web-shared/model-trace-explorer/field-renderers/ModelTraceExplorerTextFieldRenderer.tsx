import { useState } from 'react';

import { ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';

const STRING_TRUNCATION_LIMIT = 400;

export const ModelTraceExplorerTextFieldRenderer = ({ title, value }: { title: string; value: string }) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);

  const isExpandable = value.length > STRING_TRUNCATION_LIMIT;
  const displayValue =
    !expanded && value.length > STRING_TRUNCATION_LIMIT ? value.slice(0, STRING_TRUNCATION_LIMIT) + '...' : value;

  const hoverStyles = isExpandable
    ? { ':hover': { backgroundColor: theme.colors.actionIconBackgroundHover, cursor: 'pointer' } }
    : {};

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
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
            padding: `${theme.spacing.sm}px ${theme.spacing.sm + theme.spacing.xs}px`,
            ...hoverStyles,
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Typography.Title level={4} color="secondary" withoutMargins>
            {title}
          </Typography.Title>
          {isExpandable && (expanded ? <ChevronDownIcon /> : <ChevronRightIcon />)}
        </div>
      )}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          paddingLeft: theme.spacing.sm + theme.spacing.xs,
          paddingRight: theme.spacing.sm + theme.spacing.xs,
          paddingTop: title ? 0 : theme.spacing.sm,
          paddingBottom: theme.spacing.sm,
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

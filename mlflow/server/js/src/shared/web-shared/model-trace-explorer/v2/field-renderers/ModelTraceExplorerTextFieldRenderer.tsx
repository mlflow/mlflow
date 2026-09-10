import { useState } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { GenAIMarkdownRenderer } from '../../../genai-markdown-renderer/GenAIMarkdownRenderer';
import { CopyActionButton } from '../../../copy/CopyActionButton';

const STRING_TRUNCATION_LIMIT = 400;

export const ModelTraceExplorerTextFieldRenderer = ({
  title,
  value,
}: {
  title: string;
  value: string;
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);

  const isExpandable = value.length > STRING_TRUNCATION_LIMIT;
  const displayValue =
    !expanded && value.length > STRING_TRUNCATION_LIMIT ? value.slice(0, STRING_TRUNCATION_LIMIT) + '...' : value;

  return (
    <div
      css={{
        borderRadius: theme.borders.borderRadiusSm,
        position: 'relative',
      }}
    >
      {title && (
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingInline: 0,
            paddingRight: theme.spacing.lg,
            marginBottom: theme.spacing.xs,
          }}
        >
          <Typography.Text bold color="secondary" size="sm">
            {title}
          </Typography.Text>
        </div>
      )}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          marginInline: 0,
          paddingLeft: 0,
          paddingRight: theme.spacing.lg,
          paddingBlock: theme.spacing.sm,
          border: 'none',
          borderRadius: 0,
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
      <CopyActionButton
        componentId="shared.model-trace-explorer.copy-scalar"
        copyText={value}
        buttonProps={{
          style: {
            position: 'absolute',
            top: title ? 0 : theme.spacing.xs,
            right: 0,
          },
        }}
      />
    </div>
  );
};

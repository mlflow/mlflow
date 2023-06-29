import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ICellRendererParams } from '@ag-grid-community/core';
import { FormattedMessage } from 'react-intl';
import React from 'react';

// Truncate the text in the cell, it doesn't make sense to populate
// more data into the DOM since cells have hidden overflow anyway
const MAX_TEXT_LENGTH = 512;

interface EvaluationTextCellRendererProps extends ICellRendererParams {
  value: string;
  highlightEnabled?: boolean;
  context: { highlightedText: string };
}

/**
 * Internal use component - breaks down the rendered text into chunks and highlights
 * particular part found by the provided substring.
 */
const HighlightedText = React.memo(({ text, highlight }: { text: string; highlight: string }) => {
  const { theme } = useDesignSystemTheme();
  if (!highlight) {
    return <>{text}</>;
  }

  const parts = text.split(new RegExp(`(${highlight})`, 'gi'));

  return (
    <>
      {parts.map((part, i) => (
        <React.Fragment key={i}>
          {part.toLowerCase() === highlight.toLowerCase() ? (
            <span css={{ backgroundColor: theme.colors.yellow200 }}>{part}</span>
          ) : (
            part
          )}
        </React.Fragment>
      ))}
    </>
  );
});

/**
 * Component used to render a single text cell in the evaluation artifacts comparison table.
 */
export const EvaluationTextCellRenderer = ({
  value,
  context,
  highlightEnabled,
}: EvaluationTextCellRendererProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        height: '100%',
        whiteSpace: 'normal',
        padding: theme.spacing.sm,
        overflow: 'hidden',
        position: 'relative',
        cursor: 'pointer',
        '&:hover': {
          backgroundColor: theme.colors.actionDefaultBackgroundHover,
        },
      }}
    >
      {!value ? (
        <Typography.Text color='info' css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
          <FormattedMessage
            defaultMessage='(empty)'
            description='Experiment page > artifact compare view > results table > no result (empty cell)'
          />
        </Typography.Text>
      ) : (
        <span
          css={{
            display: '-webkit-box',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            '-webkit-box-orient': 'vertical',
            '-webkit-line-clamp': '9',
          }}
        >
          {highlightEnabled && context.highlightedText ? (
            <HighlightedText text={value} highlight={context.highlightedText} />
          ) : (
            value.substring(0, MAX_TEXT_LENGTH)
          )}
        </span>
      )}
    </div>
  );
};

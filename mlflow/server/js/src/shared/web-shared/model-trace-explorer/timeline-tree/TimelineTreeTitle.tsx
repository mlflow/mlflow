import React from 'react';

import { Tag, Typography, useDesignSystemTheme, XCircleIcon } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';

export const TimelineTreeTitle = ({
  node,
  spanTimeFormatter,
  withTimePill,
}: {
  node: ModelTraceSpanNode;
  spanTimeFormatter: (us: number) => string;
  withTimePill: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const hasException = getSpanExceptionCount(node) > 0;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        flexDirection: 'row',
        height: theme.typography.lineHeightBase,
      }}
    >
      <Typography.Text>{node.title}</Typography.Text>
      {withTimePill && (
        <Tag
          color={hasException ? 'coral' : 'default'}
          componentId="shared.model-trace-explorer.timeline-tree-title-time-pill"
          title={spanTimeFormatter(node.end - node.start)}
          css={{
            margin: 0,
            marginLeft: theme.spacing.sm,
            fontSize: theme.typography.fontSizeSm,
          }}
        >
          {hasException && <XCircleIcon css={{ marginRight: theme.spacing.xs, fontSize: 14, color: 'inherit' }} />}
          <span css={{ fontSize: theme.typography.fontSizeSm }}>{spanTimeFormatter(node.end - node.start)}</span>
        </Tag>
      )}
    </div>
  );
};

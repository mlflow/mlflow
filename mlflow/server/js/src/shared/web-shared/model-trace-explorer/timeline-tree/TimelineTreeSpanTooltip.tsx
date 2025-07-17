import React from 'react';

import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { spanTimeFormatter } from './TimelineTree.utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { getIconTypeForSpan } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';

export const TimelineTreeSpanTooltip = ({
  span,
  children,
}: {
  span: ModelTraceSpanNode;
  children: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const iconType = getIconTypeForSpan(span.type ?? ModelSpanType.UNKNOWN);
  const primaryTextColor = theme.isDarkMode ? theme.colors.grey800 : theme.colors.grey100;
  const secondaryTextColor = theme.isDarkMode ? theme.colors.grey500 : theme.colors.grey350;

  return (
    <Tooltip
      componentId="shared.model-trace-explorer.timeline-tree-node-tooltip"
      hideWhenDetached={false}
      content={
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              gap: theme.spacing.xs,
              alignItems: 'center',
              overflow: 'hidden',
              wordBreak: 'break-all',
            }}
          >
            <ModelTraceExplorerIcon type={iconType} isInTooltip />
            <span css={{ color: primaryTextColor }}>{span.title}</span>
            <span
              css={{ marginLeft: theme.spacing.xs, color: secondaryTextColor, fontSize: theme.typography.fontSizeSm }}
            >
              {spanTimeFormatter(span.end - span.start)}
            </span>
          </div>
          <div css={{ display: 'flex', flexDirection: 'row', color: primaryTextColor }}>
            <FormattedMessage defaultMessage="Start:" description="Label for the start time of a span" />{' '}
            {spanTimeFormatter(span.start)}
            {' | '}
            <FormattedMessage defaultMessage="End:" description="Label for the end time of a span" />{' '}
            {spanTimeFormatter(span.end)}
          </div>
        </div>
      }
      side="right"
      maxWidth={700}
    >
      {children}
    </Tooltip>
  );
};

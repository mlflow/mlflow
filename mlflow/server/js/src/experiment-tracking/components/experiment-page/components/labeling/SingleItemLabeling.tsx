/**
 * Single item labeling view - displays a trace with labeling tasks
 *
 * Simplified for OSS - shows trace data and labeling tasks side-by-side.
 */

import React from 'react';
import { useDesignSystemTheme, Typography } from '@databricks/design-system';

import type { LabelingSession, LabelingItem } from '../../../types/labeling';
import type { Assessment } from '../../../../shared/web-shared/model-trace-explorer/ModelTrace.types';
import { LabelingContextProvider } from './LabelingContextProvider';
import { LabelingTasks } from './LabelingTasks';

interface SingleItemLabelingProps {
  session: LabelingSession;
  item: LabelingItem;
  assessments: Assessment[];
}

export const SingleItemLabeling = ({ session, item, assessments }: SingleItemLabelingProps) => {
  const { theme } = useDesignSystemTheme();

  if (!item.trace_id) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Text>No trace associated with this labeling item.</Typography.Text>
      </div>
    );
  }

  return (
    <LabelingContextProvider session={session} item={item}>
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: theme.spacing.lg,
          padding: theme.spacing.lg,
          height: '100%',
        }}
      >
        {/* Left: Trace information */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <Typography.Title level={3}>Trace</Typography.Title>
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.legacyBorders.borderRadiusLg,
              padding: theme.spacing.md,
            }}
          >
            <Typography.Text>Trace ID: {item.trace_id}</Typography.Text>
            {/* TODO: Add trace viewer component */}
            <div css={{ marginTop: theme.spacing.md }}>
              <Typography.Paragraph>
                Trace details will be displayed here.
              </Typography.Paragraph>
            </div>
          </div>
        </div>

        {/* Right: Labeling tasks */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <Typography.Title level={3}>Labeling Tasks</Typography.Title>
          <LabelingTasks
            session={session}
            traceId={item.trace_id}
            assessments={assessments}
          />
        </div>
      </div>
    </LabelingContextProvider>
  );
};

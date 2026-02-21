/**
 * Labeling session page - main page for a labeling session
 *
 * Displays the list of items in a session and allows navigation between them.
 */

import React from 'react';
import { useParams } from 'react-router-dom';
import { useDesignSystemTheme, Typography, Spinner, Button } from '@databricks/design-system';

import { useGetLabelingSession, useGetLabelingItems } from '../../../sdk/labeling';
import { SingleItemLabeling } from './SingleItemLabeling';

export const LabelingSessionPage = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId, sessionId } = useParams<{ experimentId: string; sessionId: string }>();

  const sessionQuery = useGetLabelingSession(sessionId ?? '', { enabled: Boolean(sessionId) });
  const itemsQuery = useGetLabelingItems(
    { labeling_session_id: sessionId ?? '' },
    { enabled: Boolean(sessionId) },
  );

  const [currentItemIndex, setCurrentItemIndex] = React.useState(0);

  if (sessionQuery.isLoading || itemsQuery.isLoading) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.xl }}>
        <Spinner />
      </div>
    );
  }

  if (sessionQuery.error || itemsQuery.error) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Text>Error loading labeling session.</Typography.Text>
      </div>
    );
  }

  const session = sessionQuery.data?.labeling_session;
  const items = itemsQuery.data?.labeling_items ?? [];

  if (!session) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Text>Labeling session not found.</Typography.Text>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Title level={2}>{session.name}</Typography.Title>
        <Typography.Text>No items in this labeling session.</Typography.Text>
      </div>
    );
  }

  const currentItem = items[currentItemIndex];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <div
        css={{
          borderBottom: `1px solid ${theme.colors.border}`,
          padding: theme.spacing.md,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div>
          <Typography.Title level={2}>{session.name}</Typography.Title>
          <Typography.Text>
            Item {currentItemIndex + 1} of {items.length}
          </Typography.Text>
        </div>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.labeling.previous"
            type="tertiary"
            disabled={currentItemIndex === 0}
            onClick={() => setCurrentItemIndex(currentItemIndex - 1)}
          >
            Previous
          </Button>
          <Button
            componentId="mlflow.labeling.next"
            type="primary"
            disabled={currentItemIndex === items.length - 1}
            onClick={() => setCurrentItemIndex(currentItemIndex + 1)}
          >
            Next
          </Button>
        </div>
      </div>

      {/* Content */}
      <div css={{ flex: 1, overflow: 'auto' }}>
        <SingleItemLabeling session={session} item={currentItem} assessments={[]} />
      </div>
    </div>
  );
};

/**
 * Labeling sessions tab page - shows list of labeling sessions for an experiment
 */

import React from 'react';
import { useParams } from '../../../common/utils/RoutingUtils';
import { useDesignSystemTheme, Typography, Spinner, Button } from '@databricks/design-system';

import { useGetLabelingSessions } from '../../sdk/labeling';

export const ExperimentPageLabelingSessionsTab = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams<{ experimentId: string }>();

  const sessionsQuery = useGetLabelingSessions(experimentId ?? '', {
    enabled: Boolean(experimentId),
  });

  if (sessionsQuery.isLoading) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.xl }}>
        <Spinner />
      </div>
    );
  }

  if (sessionsQuery.error) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Text>Error loading labeling sessions.</Typography.Text>
      </div>
    );
  }

  const sessions = sessionsQuery.data?.labeling_sessions ?? [];

  if (sessions.length === 0) {
    return (
      <div css={{ padding: theme.spacing.lg, textAlign: 'center' }}>
        <Typography.Title level={3}>No Labeling Sessions</Typography.Title>
        <Typography.Paragraph>
          Create a labeling session to start labeling traces in this experiment.
        </Typography.Paragraph>
        <Button componentId="mlflow.labeling.create-session" type="primary">
          Create Labeling Session
        </Button>
      </div>
    );
  }

  return (
    <div css={{ padding: theme.spacing.lg }}>
      <div css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Title level={2}>Labeling Sessions</Typography.Title>
        <Button componentId="mlflow.labeling.create-session" type="primary">
          Create Session
        </Button>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {sessions.map((session) => (
          <div
            key={session.labeling_session_id}
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.legacyBorders.borderRadiusLg,
              padding: theme.spacing.md,
            }}
          >
            <Typography.Title level={4}>{session.name}</Typography.Title>
            <Typography.Text>
              Created: {new Date(session.creation_time).toLocaleString()}
            </Typography.Text>
            <div css={{ marginTop: theme.spacing.sm }}>
              <Typography.Text>{session.labelingSchemas.length} schemas</Typography.Text>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

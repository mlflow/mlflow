import { Typography } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import type { ExperimentEntity } from '../../../types';
import { ExperimentViewArtifactLocation } from './ExperimentViewArtifactLocation';

export const ExperimentViewDescriptions = React.memo(({ experiment }: { experiment: ExperimentEntity }) => (
  <div css={styles.container}>
    <Typography.Text color="secondary">
      <FormattedMessage
        defaultMessage="Experiment ID"
        description="Label for displaying the current experiment in view"
      />
      : {experiment.experimentId}
    </Typography.Text>
    <Typography.Text color="secondary">
      <FormattedMessage
        defaultMessage="Artifact Location"
        description="Label for displaying the experiment artifact location"
      />
      : <ExperimentViewArtifactLocation artifactLocation={experiment.artifactLocation} />
    </Typography.Text>
  </div>
));

const styles = {
  container: (theme: Theme) => ({
    display: 'flex' as const,
    gap: theme.spacing.lg,
  }),
};

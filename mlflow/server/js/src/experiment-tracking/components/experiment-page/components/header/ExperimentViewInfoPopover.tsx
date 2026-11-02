import { InfoPopover, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ExperimentEntity } from '../../../../types';
import { ExperimentViewArtifactLocation } from '../ExperimentViewArtifactLocation';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import { ExperimentViewCopyExperimentId } from './ExperimentViewCopyExperimentId';
import { ExperimentViewCopyArtifactLocation } from './ExperimentViewCopyArtifactLocation';
import { formatTraceArchivalRetentionForDisplay } from '../../../../../common/utils/traceArchival';

/**
 * The experiment's path, ID, artifact location and trace archival retention, behind an info icon.
 */
export const ExperimentViewInfoPopover = ({ experiment }: { experiment: ExperimentEntity }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const traceArchivalRetention =
    formatTraceArchivalRetentionForDisplay(experiment.effectiveTraceArchivalRetention?.trim(), intl) || undefined;

  return (
    <InfoPopover iconTitle="Info">
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          flexWrap: 'nowrap',
        }}
        data-testid="experiment-view-header-info-tooltip-content"
      >
        <div style={{ whiteSpace: 'nowrap' }}>
          <FormattedMessage defaultMessage="Path" description="Label for displaying the current experiment path" />:{' '}
          {experiment.name + ' '}
          <ExperimentViewCopyTitle experiment={experiment} size="md" />
        </div>
        <div style={{ whiteSpace: 'nowrap' }}>
          <FormattedMessage
            defaultMessage="Experiment ID"
            description="Label for displaying the current experiment in view"
          />
          : {experiment.experimentId + ' '}
          <ExperimentViewCopyExperimentId experiment={experiment} />
        </div>
        <div style={{ whiteSpace: 'nowrap' }}>
          <FormattedMessage
            defaultMessage="Artifact Location"
            description="Label for displaying the experiment artifact location"
          />
          : <ExperimentViewArtifactLocation artifactLocation={experiment.artifactLocation} />{' '}
          <ExperimentViewCopyArtifactLocation experiment={experiment} />
        </div>
        {traceArchivalRetention && (
          <div style={{ whiteSpace: 'nowrap' }}>
            <FormattedMessage
              defaultMessage="Trace Archival Retention"
              description="Label for displaying the experiment trace archival retention"
            />
            : {traceArchivalRetention}
          </div>
        )}
      </div>
    </InfoPopover>
  );
};

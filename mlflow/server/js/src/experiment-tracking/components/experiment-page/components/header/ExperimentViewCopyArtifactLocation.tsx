import { CopyActionButton } from '@databricks/web-shared/copy';
import { useIntl } from 'react-intl';
import type { ExperimentEntity } from '../../../../types';

/**
 * Experiment page header part responsible for copying
 * the artifact location after clicking on the icon
 */
export const ExperimentViewCopyArtifactLocation = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();

  return (
    <CopyActionButton
      componentId="mlflow.experiment.header.copy-artifact-location"
      copyText={experiment.artifactLocation}
      copyTooltip={intl.formatMessage({
        defaultMessage: 'Copy artifact location',
        description: 'Copy tooltip to copy experiment artifact location from experiment runs table header',
      })}
    />
  );
};

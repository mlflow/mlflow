import { CopyIcon, Typography } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { ExperimentEntity } from '../../../../types';

/**
 * Experiment page header part responsible for copying
 * the artifact location after clicking on the icon
 */
export const ExperimentViewCopyArtifactLocation = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();

  return (
    <Typography.Text
      size="md"
      dangerouslySetAntdProps={{
        copyable: {
          text: experiment.artifactLocation,
          icon: <CopyIcon />,
          tooltips: [
            intl.formatMessage({
              defaultMessage: 'Copy artifact location',
              description: 'Copy tooltip to copy experiment artifact location from experiment runs table header',
            }),
            intl.formatMessage({
              defaultMessage: 'Artifact location copied',
              description: 'Tooltip displayed after experiment artifact location was successfully copied to clipboard',
            }),
          ],
        },
      }}
    />
  );
};

import { CopyIcon, Typography } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { ExperimentEntity } from '../../../../types';

/**
 * Experiment page header part responsible for copying
 * the experimentId after clicking on the icon
 */
export const ExperimentViewCopyExperimentId = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();

  return (
    <Typography.Text
      size="md"
      dangerouslySetAntdProps={{
        copyable: {
          text: experiment.experimentId,
          icon: <CopyIcon />,
          tooltips: [
            intl.formatMessage({
              defaultMessage: 'Copy experiment id',
              description: 'Copy tooltip to copy experiment id from experiment runs table header',
            }),
            intl.formatMessage({
              defaultMessage: 'Experiment id copied',
              description: 'Tooltip displayed after experiment id was successfully copied to clipboard',
            }),
          ],
        },
      }}
    />
  );
};

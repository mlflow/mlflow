import { CopyIcon, Typography } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { ExperimentEntity } from '../../../../types';

/**
 * Experiment page header part responsible for copying
 * the title after clicking on the icon
 */
export const ExperimentViewCopyTitle = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();

  return (
    <Typography.Text
      size='xl'
      dangerouslySetAntdProps={{
        copyable: {
          text: experiment.name,
          icon: <CopyIcon />,
          tooltips: [
            intl.formatMessage({
              defaultMessage: 'Copy path',
              description: 'Copy tooltip to copy experiment path from experiment runs table header',
            }),
            intl.formatMessage({
              defaultMessage: 'Path copied',
              description:
                'Tooltip displayed after experiment path was successfully copied to clipboard',
            }),
          ],
        },
      }}
    />
  );
};

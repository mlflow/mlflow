import { CopyActionButton } from '@databricks/web-shared/copy';
import { useIntl } from 'react-intl';
import type { ExperimentEntity } from '../../../../types';

/**
 * Experiment page header part responsible for copying
 * the title after clicking on the icon
 */
export const ExperimentViewCopyTitle = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();

  return (
    <CopyActionButton
      componentId="mlflow.experiment.header.copy-path"
      copyText={experiment.name}
      copyTooltip={intl.formatMessage({
        defaultMessage: 'Copy path',
        description: 'Copy tooltip to copy experiment path from experiment runs table header',
      })}
    />
  );
};

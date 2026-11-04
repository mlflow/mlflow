import { CopyActionButton } from '@databricks/web-shared/copy';
import { useIntl } from 'react-intl';
import type { ExperimentEntity } from '../../../../types';

/**
 * Experiment page header part responsible for copying
 * the experimentId after clicking on the icon
 */
export const ExperimentViewCopyExperimentId = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();

  return (
    <CopyActionButton
      componentId="mlflow.experiment.header.copy-experiment-id"
      copyText={experiment.experimentId}
      copyTooltip={intl.formatMessage({
        defaultMessage: 'Copy experiment id',
        description: 'Copy tooltip to copy experiment id from experiment runs table header',
      })}
    />
  );
};

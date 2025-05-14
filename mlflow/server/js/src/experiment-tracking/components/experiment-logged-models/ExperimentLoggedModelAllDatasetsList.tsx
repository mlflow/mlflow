import { Overflow } from '@databricks/design-system';
import { useMemo } from 'react';
import { type LoggedModelProto } from '../../types';
import { ExperimentLoggedModelDatasetButton } from './ExperimentLoggedModelDatasetButton';

export const ExperimentLoggedModelAllDatasetsList = ({
  loggedModel,
  empty,
}: {
  loggedModel: LoggedModelProto;
  empty?: React.ReactElement;
}) => {
  const uniqueDatasets = useMemo(() => {
    const allMetrics = loggedModel.data?.metrics ?? [];
    return allMetrics.reduce<{ dataset_name: string; dataset_digest: string; run_id: string | undefined }[]>(
      (aggregate, { dataset_digest, dataset_name, run_id }) => {
        if (
          dataset_name &&
          dataset_digest &&
          !aggregate.find(
            (dataset) => dataset.dataset_name === dataset_name && dataset.dataset_digest === dataset_digest,
          )
        ) {
          aggregate.push({ dataset_name, dataset_digest, run_id });
        }
        return aggregate;
      },
      [],
    );
  }, [loggedModel]);

  if (!uniqueDatasets.length) {
    return empty ?? <>-</>;
  }

  return (
    <Overflow>
      {uniqueDatasets.map(({ dataset_digest, dataset_name, run_id }) => (
        <ExperimentLoggedModelDatasetButton
          datasetName={dataset_name}
          datasetDigest={dataset_digest}
          runId={run_id ?? null}
          key={[dataset_name, dataset_digest].join('.')}
        />
      ))}
    </Overflow>
  );
};

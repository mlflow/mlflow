import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import {
  type DatasetWithRunType,
  ExperimentViewDatasetDrawer,
} from '../../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { useLazyGetRunQuery } from '../../run-page/hooks/useGetRunQuery';
import { transformDatasets as transformGraphQLResponseDatasets } from '../../run-page/hooks/useRunDetailsPageData';
import { keyBy } from 'lodash';
import type { KeyValueEntity } from '../../../../common/types';
import { ErrorLogType, ErrorName, PredefinedError } from '@databricks/web-shared/errors';
import { ErrorCodes } from '../../../../common/constants';
import { FormattedMessage } from 'react-intl';

class DatasetRunNotFoundError extends PredefinedError {
  errorLogType = ErrorLogType.UnexpectedSystemStateError;
  errorName = ErrorName.DatasetRunNotFoundError;
  isUserError = true;
  displayMessage = (
    <FormattedMessage
      defaultMessage="The run containing the dataset could not be found."
      description="Error message displayed when the run for the dataset is not found"
    />
  );
}

type ExperimentLoggedModelOpenDatasetDetailsContextType = {
  onDatasetClicked: (params: { datasetName: string; datasetDigest: string; runId: string }) => Promise<void>;
};

export const ExperimentLoggedModelOpenDatasetDetailsContext =
  createContext<ExperimentLoggedModelOpenDatasetDetailsContextType>({
    onDatasetClicked: () => Promise.resolve(),
  });

/**
 * Creates a context provider that allows opening the dataset details drawer from the logged model page.
 * Uses the `useGetRunQuery` GraphQL to fetch the run info for the dataset.
 */
export const ExperimentLoggedModelOpenDatasetDetailsContextProvider = ({ children }: { children: React.ReactNode }) => {
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();

  const [getRunInfo] = useLazyGetRunQuery();

  // Store the current promise's reject function
  const rejectCurrentPromiseFn = useRef<((reason?: any) => void) | null>(null);

  const onDatasetClicked = useCallback(
    async (params: { datasetName: string; datasetDigest: string; runId: string }) =>
      new Promise<void>((resolve, reject) => {
        // If there's a promise in flight, reject it to remove previous loading state
        rejectCurrentPromiseFn.current?.();

        return getRunInfo({
          onError: reject,
          onCompleted(data) {
            // If there's an API error in the response, reject the promise
            if (data.mlflowGetRun?.apiError) {
              // Special case: if the run is not found, show a different error message
              const error =
                data.mlflowGetRun.apiError.code === ErrorCodes.RESOURCE_DOES_NOT_EXIST
                  ? new DatasetRunNotFoundError()
                  : data.mlflowGetRun.apiError;
              reject(error);
              return;
            }
            // Transform the datasets into a format that can be used by the drawer UI
            const datasets = transformGraphQLResponseDatasets(data.mlflowGetRun?.run?.inputs?.datasetInputs);

            // Ensure that the datasets and run info are present
            if (!datasets || !data.mlflowGetRun?.run?.info) {
              resolve();
              return;
            }

            // Find the dataset that matches the dataset name and digest
            const matchingDataset = datasets?.find(
              (datasetInput) =>
                datasetInput.dataset?.digest === params.datasetDigest &&
                datasetInput.dataset.name === params.datasetName,
            );

            // If the dataset is not found, return early
            if (!matchingDataset) {
              resolve();
              return;
            }
            const { info, data: runData } = data.mlflowGetRun.run;

            // Convert tags into a dictionary for easier access
            const tagsDictionary = keyBy(runData?.tags?.filter((tag) => tag.key && tag.value) ?? [], 'key') as Record<
              string,
              KeyValueEntity
            >;

            // Open the drawer using the dataset and run info
            setIsDrawerOpen(true);
            setSelectedDatasetWithRun({
              datasetWithTags: {
                dataset: matchingDataset.dataset,
                tags: matchingDataset.tags,
              },
              runData: {
                datasets: datasets,
                runUuid: info.runUuid ?? '',
                experimentId: info.experimentId ?? '',
                runName: info.runName ?? '',
                tags: tagsDictionary,
              },
            });

            // Resolve the promise
            resolve();
            rejectCurrentPromiseFn.current = null;
          },
          variables: { data: { runId: params.runId } },
        });
      }),
    [getRunInfo],
  );

  const contextValue = useMemo(() => ({ onDatasetClicked }), [onDatasetClicked]);

  return (
    <ExperimentLoggedModelOpenDatasetDetailsContext.Provider value={contextValue}>
      {children}
      {selectedDatasetWithRun && (
        <ExperimentViewDatasetDrawer
          isOpen={isDrawerOpen}
          selectedDatasetWithRun={selectedDatasetWithRun}
          setIsOpen={setIsDrawerOpen}
          setSelectedDatasetWithRun={setSelectedDatasetWithRun}
        />
      )}
    </ExperimentLoggedModelOpenDatasetDetailsContext.Provider>
  );
};

export const useExperimentLoggedModelOpenDatasetDetails = () =>
  useContext(ExperimentLoggedModelOpenDatasetDetailsContext);

import { Global } from '@emotion/react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { ResizableBox } from 'react-resizable';
import { ExperimentViewRunsTableResizerHandle } from '../../components/experiment-page/components/runs/ExperimentViewRunsTableResizer';
import { useEffect, useMemo, useState } from 'react';
import { useParams } from '../../../common/utils/RoutingUtils';
import invariant from 'invariant';
import { ExperimentEvaluationDatasetsListTable } from './components/ExperimentEvaluationDatasetsListTable';
import { ExperimentEvaluationDatasetRecordsTable } from './components/ExperimentEvaluationDatasetRecordsTable';
import { ExperimentEvaluationDatasetsPageWrapper } from './ExperimentEvaluationDatasetsPageWrapper';
import { ExperimentEvaluationDatasetsEmptyState } from './components/ExperimentEvaluationDatasetsEmptyState';
import { useSelectedDatasetBySearchParam } from './hooks/useSelectedDatasetBySearchParam';
import { useSearchEvaluationDatasets } from './hooks/useSearchEvaluationDatasets';
import { useRegisterAssistantContext } from '@mlflow/mlflow/src/assistant';

const ExperimentEvaluationDatasetsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [tableWidth, setTableWidth] = useState(400);
  const [dragging, setDragging] = useState(false);
  const [datasetListHidden, setDatasetListHidden] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useSelectedDatasetBySearchParam();
  useRegisterAssistantContext('selectedDatasetId', selectedDatasetId);
  // searchFilter only gets updated after the user presses enter
  const [searchFilter, setSearchFilter] = useState('');

  invariant(experimentId, 'Experiment ID must be defined');

  const {
    data: datasets,
    isLoading,
    isFetching,
    error,
    refetch,
    fetchNextPage,
    hasNextPage,
  } = useSearchEvaluationDatasets({ experimentId, nameFilter: searchFilter });

  // Auto-select the first dataset when:
  // - No dataset is currently selected, OR
  // - The selected dataset is no longer in the list (e.g., was deleted / not in search)
  const datasetIds = useMemo(() => datasets?.map((d) => d.dataset_id) ?? [], [datasets]);
  useEffect(() => {
    if (datasets?.length && (!selectedDatasetId || !datasetIds.includes(selectedDatasetId))) {
      setSelectedDatasetId(datasets[0].dataset_id);
    }
  }, [datasets, selectedDatasetId, datasetIds, setSelectedDatasetId]);

  // Derive selected dataset from datasets and selectedDatasetId
  const selectedDataset = useMemo(() => {
    if (!datasets?.length || !selectedDatasetId) return undefined;
    return datasets.find((d) => d.dataset_id === selectedDatasetId);
  }, [datasets, selectedDatasetId]);

  return (
    <div css={{ display: 'flex', flexDirection: 'row', flex: 1, minHeight: '0px' }}>
      <ResizableBox
        css={{ display: 'flex', position: 'relative' }}
        style={{ flex: `0 0 ${datasetListHidden ? 0 : tableWidth}px` }}
        width={tableWidth}
        axis="x"
        resizeHandles={['e']}
        minConstraints={[250, 0]}
        handle={
          <ExperimentViewRunsTableResizerHandle
            runListHidden={datasetListHidden}
            updateRunListHidden={() => {
              setDatasetListHidden(!datasetListHidden);
            }}
          />
        }
        onResize={(event, { size }) => {
          if (datasetListHidden) {
            return;
          }
          setTableWidth(size.width);
        }}
        onResizeStart={() => !datasetListHidden && setDragging(true)}
        onResizeStop={() => setDragging(false)}
      >
        <div css={{ display: datasetListHidden ? 'none' : 'flex', flex: 1, minWidth: 0 }}>
          <ExperimentEvaluationDatasetsListTable
            experimentId={experimentId}
            datasets={datasets}
            isLoading={isLoading}
            isFetching={isFetching}
            error={error}
            refetch={refetch}
            fetchNextPage={fetchNextPage}
            hasNextPage={hasNextPage}
            selectedDatasetId={selectedDatasetId}
            setSelectedDatasetId={setSelectedDatasetId}
            searchFilter={searchFilter}
            setSearchFilter={setSearchFilter}
          />
        </div>
      </ResizableBox>
      <div
        css={{
          flex: 1,
          display: 'flex',
          borderLeft: `1px solid ${theme.colors.border}`,
          minHeight: '0px',
          overflow: 'hidden',
        }}
      >
        {!isLoading && !selectedDataset && <ExperimentEvaluationDatasetsEmptyState experimentId={experimentId} />}
        {selectedDataset && <ExperimentEvaluationDatasetRecordsTable dataset={selectedDataset} />}
      </div>
      {dragging && (
        <Global
          styles={{
            'body, :host': {
              userSelect: 'none',
            },
          }}
        />
      )}
    </div>
  );
};

const ExperimentEvaluationDatasetsPage = () => {
  return (
    <ExperimentEvaluationDatasetsPageWrapper>
      <ExperimentEvaluationDatasetsPageImpl />
    </ExperimentEvaluationDatasetsPageWrapper>
  );
};

export default ExperimentEvaluationDatasetsPage;

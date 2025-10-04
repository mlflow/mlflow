import { Global } from '@emotion/react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { ResizableBox } from 'react-resizable';
import { ExperimentViewRunsTableResizerHandle } from '../../components/experiment-page/components/runs/ExperimentViewRunsTableResizer';
import { useState } from 'react';
import { useParams } from '../../../common/utils/RoutingUtils';
import invariant from 'invariant';
import { ExperimentEvaluationDatasetsListTable } from './components/ExperimentEvaluationDatasetsListTable';
import { ExperimentEvaluationDatasetRecordsTable } from './components/ExperimentEvaluationDatasetRecordsTable';
import { EvaluationDataset } from './types';
import { ExperimentEvaluationDatasetsPageWrapper } from './ExperimentEvaluationDatasetsPageWrapper';
import { ExperimentEvaluationDatasetsEmptyState } from './components/ExperimentEvaluationDatasetsEmptyState';

const ExperimentEvaluationDatasetsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [tableWidth, setTableWidth] = useState(400);
  const [dragging, setDragging] = useState(false);
  const [datasetListHidden, setDatasetListHidden] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<EvaluationDataset | undefined>(undefined);
  const [isDatasetsLoading, setIsDatasetsLoading] = useState(false);

  invariant(experimentId, 'Experiment ID must be defined');

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
            setIsLoading={setIsDatasetsLoading}
            experimentId={experimentId}
            selectedDataset={selectedDataset}
            setSelectedDataset={setSelectedDataset}
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
        {!isDatasetsLoading && !selectedDataset && (
          <ExperimentEvaluationDatasetsEmptyState experimentId={experimentId} />
        )}
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

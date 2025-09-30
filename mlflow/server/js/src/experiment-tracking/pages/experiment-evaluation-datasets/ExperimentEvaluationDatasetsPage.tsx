import { Global } from '@emotion/react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ResizableBox } from 'react-resizable';
import { ExperimentViewRunsTableResizerHandle } from '../../components/experiment-page/components/runs/ExperimentViewRunsTableResizer';
import { useState, useCallback } from 'react';
import { useParams } from '../../../common/utils/RoutingUtils';
import invariant from 'invariant';
import { FormattedMessage } from 'react-intl';
import { useSearchEvaluationDatasets } from './hooks/useSearchEvaluationDatsets';
import { ExperimentEvaluationDatasetsListTable } from './components/ExperimentEvaluationDatasetsListTable';

const ExperimentEvaluationDatasetsPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const [tableWidth, setTableWidth] = useState(400);
  const [dragging, setDragging] = useState(false);
  const [datasetListHidden, setDatasetListHidden] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | undefined>(undefined);

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
            updateRunListHidden={(value) => {
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
        <ExperimentEvaluationDatasetsListTable
          experimentId={experimentId}
          selectedDatasetId={selectedDatasetId}
          setSelectedDatasetId={setSelectedDatasetId}
        />
      </ResizableBox>
      <div
        css={{
          flex: 1,
          borderLeft: `1px solid ${theme.colors.border}`,
          minHeight: '0px',
          overflowX: 'hidden',
        }}
      >
        TBI
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
  return <ExperimentEvaluationDatasetsPageImpl />;
};

export default ExperimentEvaluationDatasetsPage;

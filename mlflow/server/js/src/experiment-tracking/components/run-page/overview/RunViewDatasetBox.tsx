import type { KeyValueEntity, RunDatasetWithTags, RunInfoEntity } from '../../../types';
import { Button, DropdownMenu, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentViewDatasetWithContext } from '../../experiment-page/components/runs/ExperimentViewDatasetWithContext';
import { useState } from 'react';
import {
  DatasetWithRunType,
  ExperimentViewDatasetDrawer,
} from '../../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { getStableColorForRun } from '../../../utils/RunNameUtils';

/**
 * Renders single dataset, either in overview table cell or within a dropdown
 */
const DatasetEntry = ({ dataset, onClick }: { dataset: RunDatasetWithTags; onClick: () => void }) => {
  return (
    <Typography.Link
      role="link"
      css={{
        textAlign: 'left',
      }}
      onClick={onClick}
    >
      <ExperimentViewDatasetWithContext datasetWithTags={dataset} displayTextAsLink css={{ margin: 0 }} />
    </Typography.Link>
  );
};

/**
 * Displays run datasets section in run detail overview.
 */
export const RunViewDatasetBox = ({
  tags,
  runInfo,
  datasets,
}: {
  tags: Record<string, KeyValueEntity>;
  runInfo: RunInfoEntity;
  datasets: RunDatasetWithTags[];
}) => {
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType | null>(null);
  const { theme } = useDesignSystemTheme();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  if (!datasets || !datasets.length) {
    return null;
  }

  const firstDataset = datasets[0];
  const remainingDatasets = datasets.slice(1);

  const datasetClicked = (dataset: RunDatasetWithTags) => {
    setSelectedDatasetWithRun({
      datasetWithTags: dataset,
      runData: {
        experimentId: runInfo.experiment_id,
        runUuid: runInfo.run_uuid,
        runName: runInfo.run_name,
        datasets: datasets,
        tags: tags,
        color: getStableColorForRun(runInfo.run_uuid),
      },
    });
    setIsDrawerOpen(true);
  };

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
      <DatasetEntry dataset={firstDataset} onClick={() => datasetClicked(firstDataset)} />
      {remainingDatasets.length ? (
        <DropdownMenu.Root modal={false}>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdatasetbox.tsx_70"
              size="small"
            >
              +{remainingDatasets.length}
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content>
            {remainingDatasets.map((datasetWithTags) => {
              return (
                <DropdownMenu.Item key={datasetWithTags.dataset.digest}>
                  <DatasetEntry dataset={datasetWithTags} onClick={() => datasetClicked(datasetWithTags)} />
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      ) : null}
      {selectedDatasetWithRun && (
        <ExperimentViewDatasetDrawer
          isOpen={isDrawerOpen}
          setIsOpen={setIsDrawerOpen}
          selectedDatasetWithRun={selectedDatasetWithRun}
          setSelectedDatasetWithRun={setSelectedDatasetWithRun}
        />
      )}
    </div>
  );
};

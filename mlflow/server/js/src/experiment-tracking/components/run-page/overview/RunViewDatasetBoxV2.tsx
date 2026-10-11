import { Overflow, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import type { RunDatasetWithTags, RunInfoEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { DatasetWithRunType } from '../../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { ExperimentViewDatasetDrawer } from '../../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { ExperimentViewDatasetWithContext } from '../../experiment-page/components/runs/ExperimentViewDatasetWithContext';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';
import { DatasetLink } from '../../../pages/experiment-evaluation-datasets/DatasetLink';
import { parseJSONSafe } from '../../../../common/utils/TagUtils';

/**
 * Displays run datasets section in run detail overview.
 */
export const RunViewDatasetBoxV2 = ({
  tags,
  runInfo,
  datasets,
}: {
  tags: Record<string, KeyValueEntity>;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  datasets: RunDatasetWithTags[];
}) => {
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const { theme } = useDesignSystemTheme();

  if (!datasets || !datasets.length) {
    return null;
  }

  const datasetClicked = (dataset: RunDatasetWithTags) => {
    setSelectedDatasetWithRun({
      datasetWithTags: dataset,
      runData: {
        experimentId: runInfo.experimentId ?? undefined,
        runUuid: runInfo.runUuid ?? '',
        runName: runInfo.runName ?? undefined,
        datasets,
        tags,
      },
    });
    setIsDrawerOpen(true);
  };

  return (
    <>
      <Overflow>
        {datasets.map((datasetWithTags) => {
          const key = `${datasetWithTags.dataset.name}-${datasetWithTags.dataset.digest}`;
          const content = (
            <ExperimentViewDatasetWithContext datasetWithTags={datasetWithTags} displayTextAsLink={false} />
          );

          // Evaluation datasets carry a dataset_id in their source and have no schema/profile to
          // show in the drawer, so link straight to the dataset detail page (reusing the eval-runs
          // DatasetLink) instead of opening an empty drawer. Other dataset types keep the drawer.
          if (parseJSONSafe(datasetWithTags.dataset.source)?.dataset_id) {
            return (
              <DatasetLink key={key} dataset={datasetWithTags.dataset}>
                {content}
              </DatasetLink>
            );
          }

          return (
            <div
              key={key}
              role="button"
              tabIndex={0}
              css={{
                textAlign: 'left',
                cursor: 'pointer',
                '.anticon': {
                  fontSize: theme.general.iconFontSize,
                },
                '&:hover': {
                  color: theme.colors.actionPrimaryBackgroundDefault,
                },
              }}
              onClick={() => datasetClicked(datasetWithTags)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  datasetClicked(datasetWithTags);
                }
              }}
            >
              {content}
            </div>
          );
        })}
      </Overflow>
      {selectedDatasetWithRun && (
        <ExperimentViewDatasetDrawer
          isOpen={isDrawerOpen}
          setIsOpen={setIsDrawerOpen}
          selectedDatasetWithRun={selectedDatasetWithRun}
          setSelectedDatasetWithRun={setSelectedDatasetWithRun}
        />
      )}
    </>
  );
};

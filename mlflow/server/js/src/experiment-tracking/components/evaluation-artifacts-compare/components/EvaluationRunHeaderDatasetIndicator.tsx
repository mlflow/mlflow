import {
  Button,
  Popover,
  TableIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ExperimentViewDatasetWithContext } from '../../experiment-page/components/runs/ExperimentViewDatasetWithContext';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import type { RunDatasetWithTags } from '../../../types';
import { useCallback } from 'react';
import { FormattedMessage } from 'react-intl';

interface EvaluationRunHeaderDatasetIndicatorProps {
  run: RunRowType;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
}

export const EvaluationRunHeaderDatasetIndicator = ({
  run,
  onDatasetSelected,
}: EvaluationRunHeaderDatasetIndicatorProps) => {
  const { theme } = useDesignSystemTheme();

  const handleDatasetSelected = useCallback(
    (datasetWithTags: RunDatasetWithTags) => onDatasetSelected(datasetWithTags, run),
    [onDatasetSelected, run],
  );

  return (
    <div
      css={{
        flex: '1 0',
        padding: '0 8px',
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        overflow: 'hidden',
      }}
    >
      {run.datasets?.length > 0 ? (
        <>
          <div css={{ flexShrink: 1, flexGrow: 1, overflow: 'hidden' }}>
            <Button type='link' onClick={() => handleDatasetSelected(run.datasets[0])}>
              <ExperimentViewDatasetWithContext
                datasetWithTags={run.datasets[0]}
                displayTextAsLink
              />
            </Button>
          </div>
          {run.datasets.length > 1 && (
            <div css={{ flexShrink: 0, flexGrow: 1, display: 'flex', alignItems: 'flex-end' }}>
              <Popover.Root modal={false}>
                <Popover.Trigger asChild>
                  <Button size='small' style={{ borderRadius: '8px', width: '40px' }}>
                    <Typography.Text color='secondary'>+{run.datasets.length - 1}</Typography.Text>
                  </Button>
                </Popover.Trigger>
                <Popover.Content align='start'>
                  {run.datasets.slice(1).map((datasetWithTags) => (
                    <div
                      css={{
                        height: theme.general.heightSm,
                        display: 'flex',
                        alignItems: 'center',
                      }}
                      key={`${datasetWithTags.dataset.name}-${datasetWithTags.dataset.digest}`}
                    >
                      <Button type='link' onClick={() => handleDatasetSelected(datasetWithTags)}>
                        <ExperimentViewDatasetWithContext
                          datasetWithTags={datasetWithTags}
                          displayTextAsLink
                        />
                      </Button>
                    </div>
                  ))}
                </Popover.Content>
              </Popover.Root>
            </div>
          )}
        </>
      ) : (
        <Typography.Text size='md' color='info'>
          <TableIcon css={{ marginRight: theme.spacing.xs, color: theme.colors.textSecondary }} />
          <FormattedMessage
            defaultMessage='No datasets recorded'
            description='Experiment page > artifact compare view > run header > datasets indicator > empty'
          />
        </Typography.Text>
      )}
    </div>
  );
};

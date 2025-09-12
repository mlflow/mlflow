import { throttle } from 'lodash';
import { Button, Popover, TableIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import React, { useRef, useEffect, useState, useMemo } from 'react';
import { MLFLOW_RUN_DATASET_CONTEXT_TAG } from '../../../../../constants';
import type { RunDatasetWithTags } from '../../../../../types';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import { EXPERIMENT_RUNS_TABLE_ROW_HEIGHT } from '../../../utils/experimentPage.common-utils';
import type { SuppressKeyboardEventParams } from '@ag-grid-community/core';
const MAX_DATASETS_VISIBLE = 3;

/**
 * Local component, used to render a single dataset within a cell
 * or a context menu
 */
const SingleDataset = ({
  datasetWithTags,
  onDatasetSelected,
  appendComma = false,
  inPopover = false,
}: {
  datasetWithTags: RunDatasetWithTags;
  onDatasetSelected: () => void;
  appendComma?: boolean;
  inPopover?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const { dataset, tags } = datasetWithTags;
  if (!dataset) {
    return null;
  }
  const contextTag = tags?.find(({ key }) => key === MLFLOW_RUN_DATASET_CONTEXT_TAG);
  return (
    <div
      css={{
        display: 'flex',
        flexShrink: 0,
        alignItems: 'center',
        overflow: 'hidden',
        marginRight: theme.spacing.xs,
      }}
    >
      <TableIcon css={{ color: theme.colors.textSecondary, marginRight: theme.spacing.xs }} />{' '}
      <span
        css={{ minWidth: 32, marginRight: theme.spacing.xs, flexShrink: 0 }}
        title={`${dataset.name} (${dataset.digest})`}
      >
        {inPopover ? (
          <Popover.Close asChild>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_49"
              type="link"
              onClick={onDatasetSelected}
              tabIndex={0}
            >
              <span css={{ fontSize: 12 }}>
                {dataset.name} ({dataset.digest})
              </span>
            </Button>
          </Popover.Close>
        ) : (
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_56"
            type="link"
            onClick={onDatasetSelected}
            data-testid="open-dataset-drawer"
            tabIndex={0}
          >
            <span>
              {dataset.name} ({dataset.digest})
            </span>
          </Button>
        )}
      </span>
      {contextTag && (
        <Tag
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_75"
          css={{ textTransform: 'capitalize', marginRight: theme.spacing.xs }}
        >
          <span css={{ fontSize: 12 }}>{contextTag.value}</span>
        </Tag>
      )}
      {appendComma && <>,</>}
    </div>
  );
};

export interface DatasetsCellRendererProps {
  value: RunDatasetWithTags[];
  data: RunRowType;
  onDatasetSelected: (dataset: RunDatasetWithTags, run: RunRowType) => void;
  expandRows: boolean;
}

export const DatasetsCellRenderer = React.memo(
  ({ value: datasets, data, onDatasetSelected, expandRows }: DatasetsCellRendererProps) => {
    const containerElement = useRef<HTMLDivElement>(null);
    const [datasetsVisible, setDatasetsVisible] = useState(0);
    const [ellipsisVisible, setEllipsisVisible] = useState(false);
    const clampedDatasets = useMemo(() => (datasets || []).slice(0, MAX_DATASETS_VISIBLE), [datasets]);
    const { theme } = useDesignSystemTheme();

    const datasetsLength = (datasets || []).length;

    useEffect(() => {
      if (!containerElement.current) {
        return () => {};
      }
      const callback: ResizeObserverCallback = throttle(([entry]) => {
        if (expandRows) {
          const availableHeight = entry.contentRect.height;
          let elementsFit = 0;
          let stackedHeight = 0;
          for (let i = 0; i < entry.target.children.length; i++) {
            const item = entry.target.children.item(i) as Element;
            if (stackedHeight + item.clientHeight > availableHeight) {
              break;
            }
            stackedHeight += item.clientHeight;
            elementsFit++;
          }
          setDatasetsVisible(elementsFit);
          setEllipsisVisible(elementsFit < datasetsLength);
        } else {
          const availableWidth = entry.contentRect.width;
          if (availableWidth === 0 && datasetsLength) {
            setDatasetsVisible(0);
            setEllipsisVisible(true);
            return;
          }
          let elementsFit = 0;
          let stackedWidth = 0;
          for (let i = 0; i < entry.target.children.length; i++) {
            const item = entry.target.children.item(i) as Element;
            if (stackedWidth + item.clientWidth >= availableWidth) {
              break;
            }
            stackedWidth += item.clientWidth;
            elementsFit++;
          }
          const partiallyVisibleDatasets = Math.min(datasetsLength, elementsFit + 1);
          setDatasetsVisible(partiallyVisibleDatasets);
          setEllipsisVisible(elementsFit < datasetsLength);
        }
      }, 100);

      const resizeObserver = new ResizeObserver(callback);

      resizeObserver.observe(containerElement.current);
      return () => resizeObserver.disconnect();
    }, [expandRows, datasetsLength]);

    const moreItemsToShow = datasetsLength - datasetsVisible;
    if (!datasets || datasetsLength < 1) {
      return <>-</>;
    }

    const datasetsToShow = expandRows ? clampedDatasets : datasets;

    return (
      <div css={{ display: 'flex' }}>
        <div
          css={{
            overflow: 'hidden',
            display: 'flex',
            flexDirection: expandRows ? 'column' : 'row',
          }}
          ref={containerElement}
        >
          {datasetsToShow.map((datasetWithTags, index) => (
            <SingleDataset
              appendComma={expandRows ? false : index < datasetsToShow.length - 1}
              key={`${datasetWithTags.dataset.name}-${datasetWithTags.dataset.digest}`}
              datasetWithTags={datasetWithTags}
              onDatasetSelected={() => onDatasetSelected?.(datasetWithTags, data)}
            />
          ))}
        </div>
        {(moreItemsToShow > 0 || ellipsisVisible) && (
          <div css={{ display: 'flex', alignItems: 'flex-end' }}>
            {!expandRows && ellipsisVisible && (
              <span css={{ paddingLeft: 0, paddingRight: theme.spacing.xs }}>&hellip;</span>
            )}
            {moreItemsToShow > 0 && (
              <Popover.Root
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_184"
                modal={false}
              >
                <Popover.Trigger asChild>
                  <Button
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_datasetscellrenderer.tsx_172"
                    size="small"
                    style={{ borderRadius: '8px', width: '40px' }}
                    tabIndex={0}
                  >
                    <Typography.Text color="secondary">+{moreItemsToShow}</Typography.Text>
                  </Button>
                </Popover.Trigger>
                <Popover.Content align="start" css={{ maxHeight: '400px', overflow: 'auto' }}>
                  {datasets.slice(datasetsLength - moreItemsToShow).map((datasetWithTags) => (
                    <div
                      css={{
                        height: theme.general.heightSm,
                        display: 'flex',
                        alignItems: 'center',
                      }}
                      key={`${datasetWithTags.dataset.name}-${datasetWithTags.dataset.digest}`}
                    >
                      <SingleDataset
                        datasetWithTags={datasetWithTags}
                        onDatasetSelected={() => onDatasetSelected?.(datasetWithTags, data)}
                        inPopover
                      />
                    </div>
                  ))}
                </Popover.Content>
              </Popover.Root>
            )}
          </div>
        )}
      </div>
    );
  },
);

export const getDatasetsCellHeight = (datasetColumnShown: boolean, row: { data: RunRowType }) => {
  if (datasetColumnShown) {
    const { data } = row;

    // Display at least 1, but at most 5 text lines in the cell.
    const datasetsCount = Math.min(data.datasets?.length || 1, MAX_DATASETS_VISIBLE);
    return EXPERIMENT_RUNS_TABLE_ROW_HEIGHT * datasetsCount;
  }
  return EXPERIMENT_RUNS_TABLE_ROW_HEIGHT;
};

/**
 * A utility function that enables custom keyboard navigation for the datasets cell renderer by providing
 * conditional suppression of default events.
 *
 * This cell needs specific handling since it's the only one that displays multiple buttons simultaneously.
 */
export const DatasetsCellRendererSuppressKeyboardEvents = ({ event }: SuppressKeyboardEventParams) => {
  return (
    event.key === 'Tab' &&
    event.target instanceof HTMLElement &&
    // Let's suppress the default action if the focus is on cell or on the dataset button, allowing
    // tab to move to the next focusable element.
    (event.target.classList.contains('ag-cell') || event.target instanceof HTMLButtonElement)
  );
};

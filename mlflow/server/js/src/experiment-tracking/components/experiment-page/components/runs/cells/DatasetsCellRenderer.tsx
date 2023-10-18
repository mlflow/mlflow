import { throttle } from 'lodash';
import {
  Button,
  Popover,
  TableIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import React, { useRef, useEffect, useState, useMemo } from 'react';
import {
  ATTRIBUTE_COLUMN_LABELS,
  COLUMN_TYPES,
  MLFLOW_RUN_DATASET_CONTEXT_TAG,
} from '../../../../../constants';
import type { RunDatasetWithTags } from '../../../../../types';
import { RunRowType } from '../../../utils/experimentPage.row-types';
import { makeCanonicalSortKey } from '../../../utils/experimentPage.column-utils';
import { shouldEnableExperimentDatasetTracking } from '../../../../../../common/utils/FeatureUtils';
import { EXPERIMENT_RUNS_TABLE_ROW_HEIGHT } from '../../../utils/experimentPage.common-utils';
import { SearchExperimentRunsFacetsState } from 'experiment-tracking/components/experiment-page/models/SearchExperimentRunsFacetsState';
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
      <span css={{ minWidth: 32, marginRight: theme.spacing.xs, flexShrink: 0 }}>
        {inPopover ? (
          <Popover.Close asChild>
            <Button type='link' onClick={onDatasetSelected}>
              <span css={{ fontSize: 12 }}>
                {dataset.name} ({dataset.digest})
              </span>
            </Button>
          </Popover.Close>
        ) : (
          <Button type='link' onClick={onDatasetSelected}>
            <span>
              {dataset.name} ({dataset.digest})
            </span>
          </Button>
        )}
      </span>
      {contextTag && (
        <Tag css={{ textTransform: 'capitalize', marginRight: theme.spacing.xs }}>
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
  ({ value, data, onDatasetSelected, expandRows }: DatasetsCellRendererProps) => {
    const containerElement = useRef<HTMLDivElement>(null);
    const [datasetsVisible, setDatasetsVisible] = useState(0);
    const clampedDatasets = useMemo(() => value.slice(0, MAX_DATASETS_VISIBLE), [value]);
    const { theme } = useDesignSystemTheme();

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
        } else {
          const availableWidth = entry.contentRect.width;
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
          setDatasetsVisible(elementsFit);
        }
      }, 100);

      const resizeObserver = new ResizeObserver(callback);

      resizeObserver.observe(containerElement.current);
      return () => resizeObserver.disconnect();
    }, [expandRows]);

    const moreItemsToShow = value.length - datasetsVisible;
    if (!value || value.length < 1) {
      return <>-</>;
    }

    const datasetsToShow = expandRows ? clampedDatasets : value;

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
              appendComma={expandRows ? false : index < clampedDatasets.length - 1}
              key={`${datasetWithTags.dataset.name}-${datasetWithTags.dataset.digest}`}
              datasetWithTags={datasetWithTags}
              onDatasetSelected={() => onDatasetSelected?.(datasetWithTags, data)}
            />
          ))}
        </div>
        {moreItemsToShow > 0 && (
          <div css={{ display: 'flex', alignItems: 'flex-end' }}>
            {!expandRows && (
              <span css={{ paddingLeft: theme.spacing.xs, paddingRight: theme.spacing.xs }}>
                &hellip;
              </span>
            )}
            <Popover.Root modal={false}>
              <Popover.Trigger asChild>
                <Button size='small' style={{ borderRadius: '8px', width: '40px' }}>
                  <Typography.Text color='secondary'>+{moreItemsToShow}</Typography.Text>
                </Button>
              </Popover.Trigger>
              <Popover.Content align='start' css={{ maxHeight: '400px', overflow: 'auto' }}>
                {value.slice(value.length - moreItemsToShow).map((datasetWithTags) => (
                  <div
                    css={{ height: theme.general.heightSm, display: 'flex', alignItems: 'center' }}
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
          </div>
        )}
      </div>
    );
  },
);

export const getDatasetsCellHeight = (
  searchFacetsState: SearchExperimentRunsFacetsState,
  row: { data: RunRowType },
) => {
  const datasetColumnId = makeCanonicalSortKey(
    COLUMN_TYPES.ATTRIBUTES,
    ATTRIBUTE_COLUMN_LABELS.DATASET,
  );
  if (
    shouldEnableExperimentDatasetTracking() &&
    searchFacetsState.selectedColumns.includes(datasetColumnId)
  ) {
    const { data } = row;

    // Display at least 1, but at most 5 text lines in the cell.
    const datasetsCount = Math.min(data.datasets?.length || 1, MAX_DATASETS_VISIBLE);
    return EXPERIMENT_RUNS_TABLE_ROW_HEIGHT * datasetsCount;
  }
  return EXPERIMENT_RUNS_TABLE_ROW_HEIGHT;
};

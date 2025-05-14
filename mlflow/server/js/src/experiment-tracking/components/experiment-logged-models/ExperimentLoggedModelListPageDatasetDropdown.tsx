import {
  Button,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCountBadge,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  TableIcon,
  useDesignSystemTheme,
  XCircleFillIcon,
} from '@databricks/design-system';
import { useMemo, useRef } from 'react';
import type { LoggedModelMetricDataset, LoggedModelProto } from '../../types';
import { useIntl } from 'react-intl';

const getDatasetHash = (dataset: LoggedModelMetricDataset) =>
  JSON.stringify([dataset.dataset_name, dataset.dataset_digest]);

export const ExperimentLoggedModelListPageDatasetDropdown = ({
  loggedModelsData,
  selectedFilterDatasets,
  onToggleDataset,
  onClearSelectedDatasets,
}: {
  loggedModelsData: LoggedModelProto[];
  selectedFilterDatasets?: LoggedModelMetricDataset[];
  onToggleDataset?: (dataset: LoggedModelMetricDataset) => void;
  onClearSelectedDatasets?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const cachedDatasets = useRef<Map<string, { hash: string } & LoggedModelMetricDataset>>(new Map());

  // Get all datasets with their hashes, also store them in an aggregated map.
  // The hash is used as a unique key and also being fed to DialogCombobox since it exclusively uses string values.
  // The map is used to aggregate all datasets encountered in the logged models data during the session.
  const allDatasets = useMemo(() => {
    for (const model of loggedModelsData) {
      for (const metric of model.data?.metrics || []) {
        if (!metric.dataset_name || !metric.dataset_digest) {
          continue;
        }
        const datasetHash = getDatasetHash(metric);
        if (!cachedDatasets.current.has(datasetHash)) {
          // We're purposely using mutable hashmap in the memo hook
          cachedDatasets.current.set(datasetHash, {
            hash: datasetHash,
            dataset_name: metric.dataset_name,
            dataset_digest: metric.dataset_digest,
          });
        }
      }
    }
    return Array.from(cachedDatasets.current.values());
  }, [loggedModelsData]);

  // Serialize the selected datasets to a string format for the DialogCombobox.
  const serializedSelectedDatasets = useMemo(
    () => selectedFilterDatasets?.map(getDatasetHash) || [],
    [selectedFilterDatasets],
  );

  return (
    <DialogCombobox
      componentId="mlflow.logged_model.list_page.datasets_filter"
      id="mlflow.logged_model.list_page.datasets_filter"
      value={serializedSelectedDatasets}
      label={intl.formatMessage({
        defaultMessage: 'Datasets',
        description: 'Label for the datasets filter dropdown in the logged model list page',
      })}
      stayOpenOnSelection
    >
      <DialogComboboxCustomButtonTriggerWrapper>
        <Button
          endIcon={<ChevronDownIcon />}
          componentId="mlflow.logged_model.list_page.datasets_filter.toggle"
          icon={<TableIcon />}
        >
          Datasets
          {serializedSelectedDatasets.length > 0 ? (
            <>
              <DialogComboboxCountBadge css={{ marginLeft: 4 }}>
                {serializedSelectedDatasets.length}
              </DialogComboboxCountBadge>
              <XCircleFillIcon
                aria-hidden="false"
                role="button"
                onClick={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  onClearSelectedDatasets?.();
                }}
                css={{
                  color: theme.colors.textPlaceholder,
                  fontSize: theme.typography.fontSizeSm,
                  marginLeft: theme.spacing.xs,

                  ':hover': {
                    color: theme.colors.actionTertiaryTextHover,
                  },
                }}
              />
            </>
          ) : null}
        </Button>
      </DialogComboboxCustomButtonTriggerWrapper>
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          {allDatasets.map(({ hash: serializedDataset, dataset_digest, dataset_name }) => (
            <DialogComboboxOptionListCheckboxItem
              value={serializedDataset}
              checked={serializedSelectedDatasets.includes(serializedDataset)}
              key={serializedDataset}
              onChange={() => onToggleDataset?.({ dataset_digest, dataset_name })}
            >
              {dataset_name} (#{dataset_digest})
            </DialogComboboxOptionListCheckboxItem>
          ))}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

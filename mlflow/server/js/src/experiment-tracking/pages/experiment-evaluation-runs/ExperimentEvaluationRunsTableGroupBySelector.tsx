import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxSectionHeader,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { EVAL_RUNS_UNSELECTABLE_COLUMNS } from './ExperimentEvaluationRunsTable.constants';
import {
  RunGroupingAggregateFunction,
  RunGroupingMode,
} from '../../components/experiment-page/utils/experimentPage.row-types';
import type { RunEntity } from '../../types';
import { FormattedMessage } from 'react-intl';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import { isGroupedBy } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import { isUserFacingTag } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { useMemo } from 'react';

export const ExperimentEvaluationRunsTableGroupBySelector = ({
  groupByConfig,
  setGroupByConfig,
  runs,
}: {
  groupByConfig: RunsGroupByConfig | null;
  setGroupByConfig: (groupBy: RunsGroupByConfig | null) => void;
  runs: RunEntity[];
}) => {
  const hasDatasets = runs.some((run) => (run.inputs?.datasetInputs?.length ?? 0) > 0);
  const { uniqueParams, uniqueTags } = useMemo(() => {
    const uniqueParams = new Set<string>();
    const uniqueTags = new Set<string>();

    for (const run of runs) {
      for (const param of run.data?.params ?? []) {
        uniqueParams.add(param.key);
      }
      for (const tag of run.data?.tags ?? []) {
        if (isUserFacingTag(tag.key)) {
          uniqueTags.add(tag.key);
        }
      }
    }

    return { uniqueParams, uniqueTags };
  }, [runs]);

  const toggleGroupBy = (mode: RunGroupingMode, columnName: string) => {
    const groupByKeys = groupByConfig?.groupByKeys ?? [];
    const newGroupByKeys = new Set(groupByKeys);
    if (isGroupedBy(groupByConfig, mode, columnName)) {
      // we can't delete objects directly from a set
      // without the reference to the original object
      for (const key of newGroupByKeys) {
        if (key.mode === mode && key.groupByData === columnName) {
          newGroupByKeys.delete(key);
        }
      }
    } else {
      newGroupByKeys.add({ mode, groupByData: columnName });
    }

    if (newGroupByKeys.size === 0) {
      setGroupByConfig(null);
    } else {
      setGroupByConfig({
        aggregateFunction: groupByConfig?.aggregateFunction ?? RunGroupingAggregateFunction.Average,
        groupByKeys: Array.from(newGroupByKeys),
      });
    }
  };

  return (
    <DialogCombobox componentId="mlflow.eval-runs.table-column-selector" label="Group by" multiSelect>
      <DialogComboboxTrigger />
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          {hasDatasets && (
            <>
              <DialogComboboxSectionHeader>
                <FormattedMessage
                  defaultMessage="Attributes"
                  description="Section header for the attributes in a 'group by' selector"
                />
              </DialogComboboxSectionHeader>
              <DialogComboboxOptionListCheckboxItem
                key="dataset"
                value="dataset"
                checked={isGroupedBy(groupByConfig, RunGroupingMode.Dataset, 'dataset')}
                onChange={() => {
                  toggleGroupBy(RunGroupingMode.Dataset, 'dataset');
                }}
              />
            </>
          )}
          {uniqueParams.size > 0 && (
            <>
              <DialogComboboxSectionHeader>
                <FormattedMessage
                  defaultMessage="Parameters"
                  description="Section header for the parameters in a 'group by' selector"
                />
              </DialogComboboxSectionHeader>
              {Array.from(uniqueParams).map((param) => (
                <DialogComboboxOptionListCheckboxItem
                  key={param}
                  value={param}
                  checked={isGroupedBy(groupByConfig, RunGroupingMode.Param, param)}
                  onChange={() => {
                    toggleGroupBy(RunGroupingMode.Param, param);
                  }}
                />
              ))}
            </>
          )}
          {uniqueTags.size > 0 && (
            <>
              <DialogComboboxSectionHeader>
                <FormattedMessage
                  defaultMessage="Tags"
                  description="Section header for the tags in a 'group by' selector"
                />
              </DialogComboboxSectionHeader>
              {Array.from(uniqueTags).map((tag) => (
                <DialogComboboxOptionListCheckboxItem
                  key={tag}
                  value={tag}
                  checked={isGroupedBy(groupByConfig, RunGroupingMode.Tag, tag)}
                  onChange={() => {
                    toggleGroupBy(RunGroupingMode.Tag, tag);
                  }}
                />
              ))}
            </>
          )}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

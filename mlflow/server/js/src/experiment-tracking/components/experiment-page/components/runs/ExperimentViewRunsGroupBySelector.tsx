import { useIntl, defineMessages, FormattedMessage } from 'react-intl';
import {
  Button,
  ChevronDownIcon,
  DropdownMenu,
  GearIcon,
  Input,
  ListBorderIcon,
  SearchIcon,
  Spinner,
  Tag,
  LegacyTooltip,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { compact, isEmpty, isString, keys, uniq, values } from 'lodash';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MLFLOW_INTERNAL_PREFIX } from '../../../../../common/utils/TagUtils';
import type { RunsGroupByConfig } from '../../utils/experimentPage.group-row-utils';
import { createRunsGroupByKey, isGroupedBy, normalizeRunsGroupByKey } from '../../utils/experimentPage.group-row-utils';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { RunGroupingAggregateFunction, RunGroupingMode } from '../../utils/experimentPage.row-types';
import { shouldEnableToggleIndividualRunsInGroups } from '../../../../../common/utils/FeatureUtils';

export interface ExperimentViewRunsGroupBySelectorProps {
  runsData: ExperimentRunsSelectorResult;
  groupBy: RunsGroupByConfig | null | string;
  onChange: (newGroupByConfig: RunsGroupByConfig | null) => void;
  useGroupedValuesInCharts?: boolean;
  onUseGroupedValuesInChartsChange: (newValue: boolean) => void;
}

const messages = defineMessages({
  minimum: {
    defaultMessage: 'Minimum',
    description: 'Experiment page > group by runs control > minimum aggregate function',
  },
  maximum: {
    defaultMessage: 'Maximum',
    description: 'Experiment page > group by runs control > maximum aggregate function',
  },
  average: {
    defaultMessage: 'Average',
    description: 'Experiment page > group by runs control > average aggregate function',
  },
  attributes: {
    defaultMessage: 'Attributes',
    description: 'Experiment page > group by runs control > attributes section label',
  },
  tags: {
    defaultMessage: 'Tags',
    description: 'Experiment page > group by runs control > tags section label',
  },
  params: {
    defaultMessage: 'Params',
    description: 'Experiment page > group by runs control > params section label',
  },
  dataset: {
    defaultMessage: 'Dataset',
    description: 'Experiment page > group by runs control > group by dataset',
  },
  noParams: {
    defaultMessage: 'No params',
    description: 'Experiment page > group by runs control > no params to group by',
  },
  noTags: {
    defaultMessage: 'No tags',
    description: 'Experiment page > group by runs control > no tags to group by',
  },
  aggregationTooltip: {
    defaultMessage: 'Aggregation: {value}',
    description: 'Experiment page > group by runs control > current aggregation function tooltip',
  },
  noResults: {
    defaultMessage: 'No results',
    description: 'Experiment page > group by runs control > no results after filtering by search query',
  },
});

const GroupBySelectorBody = ({
  runsData,
  onChange,
  groupBy,
  useGroupedValuesInCharts,
  onUseGroupedValuesInChartsChange,
}: {
  groupBy: RunsGroupByConfig;
  useGroupedValuesInCharts?: boolean;
  onChange: (newGroupBy: RunsGroupByConfig | null) => void;
  onUseGroupedValuesInChartsChange: (newValue: boolean) => void;
  runsData: ExperimentRunsSelectorResult;
}) => {
  const intl = useIntl();
  const attributeElementRef = useRef<HTMLDivElement>(null);
  const tagElementRef = useRef<HTMLDivElement>(null);
  const paramElementRef = useRef<HTMLDivElement>(null);
  const inputElementRef = useRef<any>(null);

  const minimumLabel = intl.formatMessage(messages.minimum);
  const maximumLabel = intl.formatMessage(messages.maximum);
  const averageLabel = intl.formatMessage(messages.average);
  const datasetLabel = intl.formatMessage(messages.dataset);

  const tagNames = useMemo(
    () =>
      uniq(
        values(runsData.tagsList).flatMap((runTags) =>
          keys(runTags).filter((tagKey) => !tagKey.startsWith(MLFLOW_INTERNAL_PREFIX)),
        ),
      ),
    [runsData.tagsList],
  );
  const { aggregateFunction = RunGroupingAggregateFunction.Average, groupByKeys = [] } = groupBy || {};

  const currentAggregateFunctionLabel = {
    min: minimumLabel,
    max: maximumLabel,
    average: averageLabel,
  }[aggregateFunction];

  const { theme } = useDesignSystemTheme();
  const [filter, setFilter] = useState('');

  // Autofocus won't work everywhere so let's focus input everytime the dropdown is opened
  useEffect(() => {
    requestAnimationFrame(() => {
      inputElementRef.current.focus();
    });
  }, []);

  const filteredTagNames = tagNames.filter((tag) => tag.toLowerCase().includes(filter.toLowerCase()));
  const filteredParamNames = runsData.paramKeyList.filter((param) =>
    param.toLowerCase().includes(filter.toLowerCase()),
  );
  const containsDatasets = useMemo(() => !isEmpty(compact(runsData.datasetsList)), [runsData.datasetsList]);
  const attributesMatchFilter = containsDatasets && datasetLabel.toLowerCase().includes(filter.toLowerCase());

  const hasAnyResults = filteredTagNames.length > 0 || filteredParamNames.length > 0 || attributesMatchFilter;

  const groupByToggle = useCallback(
    (mode: RunGroupingMode, groupByData: string, checked: boolean) => {
      if (checked) {
        // Scenario #1: user selected new grouping key
        const newGroupByKeys = [...groupByKeys];

        // If the key is already present, we should not add it again
        if (!newGroupByKeys.some((key) => key.mode === mode && key.groupByData === groupByData)) {
          newGroupByKeys.push({ mode, groupByData });
        }

        onChange({
          aggregateFunction,
          groupByKeys: newGroupByKeys,
        });
      } else {
        // Scenario #2: user deselected a grouping key
        const newGroupByKeys = groupByKeys.filter((key) => !(key.mode === mode && key.groupByData === groupByData));

        // If no keys are left, we should reset the group by and set it to null
        if (!newGroupByKeys.length) {
          onChange(null);
          return;
        }
        onChange({
          aggregateFunction,
          groupByKeys: newGroupByKeys,
        });
      }
    },
    [aggregateFunction, groupByKeys, onChange],
  );

  const aggregateFunctionChanged = (aggregateFunctionString: string) => {
    if (values<string>(RunGroupingAggregateFunction).includes(aggregateFunctionString)) {
      const newFunction = aggregateFunctionString as RunGroupingAggregateFunction;
      const newGroupBy: RunsGroupByConfig = { ...groupBy, aggregateFunction: newFunction };
      onChange(newGroupBy);
    }
  };

  return (
    <>
      <div css={{ display: 'flex', gap: theme.spacing.xs, padding: theme.spacing.sm }}>
        <Input
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_191"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          prefix={<SearchIcon />}
          placeholder="Search"
          autoFocus
          ref={inputElementRef}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown' || e.key === 'Tab') {
              const firstItem = attributeElementRef.current || tagElementRef.current || paramElementRef.current;
              firstItem?.focus();
              return;
            }
            if (e.key !== 'Escape') {
              e.stopPropagation();
            }
          }}
        />
        <DropdownMenu.Root>
          <LegacyTooltip
            placement="right"
            title={
              <FormattedMessage
                {...messages.aggregationTooltip}
                values={{
                  value: currentAggregateFunctionLabel || aggregateFunction,
                }}
              />
            }
          >
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_168"
                icon={<GearIcon />}
                css={{ minWidth: 32 }}
                aria-label="Change aggregation function"
              />
            </DropdownMenu.Trigger>
          </LegacyTooltip>
          <DropdownMenu.Content align="start" side="right">
            {shouldEnableToggleIndividualRunsInGroups() && (
              <>
                <DropdownMenu.CheckboxItem
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_233"
                  disabled={!groupByKeys.length}
                  checked={useGroupedValuesInCharts}
                  onCheckedChange={onUseGroupedValuesInChartsChange}
                >
                  <DropdownMenu.ItemIndicator />
                  Use grouping from the runs table in charts
                </DropdownMenu.CheckboxItem>
                <DropdownMenu.Separator />
              </>
            )}
            <DropdownMenu.RadioGroup
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_244"
              value={aggregateFunction}
              onValueChange={aggregateFunctionChanged}
            >
              <DropdownMenu.RadioItem
                disabled={!groupByKeys.length}
                value={RunGroupingAggregateFunction.Min}
                key={RunGroupingAggregateFunction.Min}
              >
                <DropdownMenu.ItemIndicator />
                {minimumLabel}
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem
                disabled={!groupByKeys.length}
                value={RunGroupingAggregateFunction.Max}
                key={RunGroupingAggregateFunction.Max}
              >
                <DropdownMenu.ItemIndicator />
                {maximumLabel}
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem
                disabled={!groupByKeys.length}
                value={RunGroupingAggregateFunction.Average}
                key={RunGroupingAggregateFunction.Average}
              >
                <DropdownMenu.ItemIndicator />
                {averageLabel}
              </DropdownMenu.RadioItem>
            </DropdownMenu.RadioGroup>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
      <DropdownMenu.Group css={{ maxHeight: 400, overflowY: 'scroll' }}>
        {attributesMatchFilter && (
          <>
            <DropdownMenu.Label>
              <FormattedMessage {...messages.attributes} />
            </DropdownMenu.Label>
            {datasetLabel.toLowerCase().includes(filter.toLowerCase()) && (
              <DropdownMenu.CheckboxItem
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_280"
                checked={isGroupedBy(groupBy, RunGroupingMode.Dataset, 'dataset')}
                key={createRunsGroupByKey(RunGroupingMode.Dataset, 'dataset', aggregateFunction)}
                ref={attributeElementRef}
                onCheckedChange={(checked) => groupByToggle(RunGroupingMode.Dataset, 'dataset', checked)}
              >
                <DropdownMenu.ItemIndicator />
                {datasetLabel}
              </DropdownMenu.CheckboxItem>
            )}
            <DropdownMenu.Separator />
          </>
        )}
        {filteredTagNames.length > 0 && (
          <>
            <DropdownMenu.Label>
              <FormattedMessage {...messages.tags} />
            </DropdownMenu.Label>

            {filteredTagNames.map((tagName, index) => {
              const groupByKey = createRunsGroupByKey(RunGroupingMode.Tag, tagName, aggregateFunction);
              return (
                <DropdownMenu.CheckboxItem
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_302"
                  checked={isGroupedBy(groupBy, RunGroupingMode.Tag, tagName)}
                  key={groupByKey}
                  ref={index === 0 ? tagElementRef : undefined}
                  onCheckedChange={(checked) => groupByToggle(RunGroupingMode.Tag, tagName, checked)}
                >
                  <DropdownMenu.ItemIndicator />
                  {tagName}
                </DropdownMenu.CheckboxItem>
              );
            })}
            {!tagNames.length && (
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_314"
                disabled
              >
                <DropdownMenu.ItemIndicator /> <FormattedMessage {...messages.noTags} />
              </DropdownMenu.Item>
            )}
            <DropdownMenu.Separator />
          </>
        )}
        {filteredParamNames.length > 0 && (
          <>
            <DropdownMenu.Label>
              <FormattedMessage {...messages.params} />
            </DropdownMenu.Label>

            {filteredParamNames.map((paramName, index) => {
              const groupByKey = createRunsGroupByKey(RunGroupingMode.Param, paramName, aggregateFunction);
              return (
                <DropdownMenu.CheckboxItem
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_330"
                  checked={isGroupedBy(groupBy, RunGroupingMode.Param, paramName)}
                  key={groupByKey}
                  ref={index === 0 ? paramElementRef : undefined}
                  onCheckedChange={(checked) => groupByToggle(RunGroupingMode.Param, paramName, checked)}
                >
                  <DropdownMenu.ItemIndicator />
                  {paramName}
                </DropdownMenu.CheckboxItem>
              );
            })}
            {!runsData.paramKeyList.length && (
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_342"
                disabled
              >
                <FormattedMessage {...messages.noParams} />
              </DropdownMenu.Item>
            )}
          </>
        )}
        {!hasAnyResults && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_349"
            disabled
          >
            <FormattedMessage {...messages.noResults} />
          </DropdownMenu.Item>
        )}
      </DropdownMenu.Group>
    </>
  );
};

/**
 * A component displaying searchable "group by" selector
 */
export const ExperimentViewRunsGroupBySelector = React.memo(
  ({
    runsData,
    groupBy,
    isLoading,
    onChange,
    useGroupedValuesInCharts,
    onUseGroupedValuesInChartsChange,
  }: ExperimentViewRunsGroupBySelectorProps & {
    isLoading: boolean;
  }) => {
    const { theme } = useDesignSystemTheme();

    // In case we encounter deprecated string-based group by descriptor
    const normalizedGroupBy = normalizeRunsGroupByKey(groupBy) || {
      aggregateFunction: RunGroupingAggregateFunction.Average,
      groupByKeys: [],
    };

    const isGroupedBy = normalizedGroupBy && !isEmpty(normalizedGroupBy.groupByKeys);

    return (
      <DropdownMenu.Root modal={false}>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_306"
            icon={<ListBorderIcon />}
            style={{ display: 'flex', alignItems: 'center' }}
            data-testid="column-selection-dropdown"
            endIcon={<ChevronDownIcon />}
          >
            {isGroupedBy ? (
              <FormattedMessage
                defaultMessage="Group by: {value}"
                description="Experiment page > group by runs control > trigger button label > with value"
                values={{
                  value: normalizedGroupBy.groupByKeys[0].groupByData,
                  // value: mode === RunGroupingMode.Dataset ? intl.formatMessage(messages.dataset) : groupByData,
                }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Group by"
                description="Experiment page > group by runs control > trigger button label > empty"
              />
            )}
            {normalizedGroupBy.groupByKeys.length > 1 && (
              <Tag
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_426"
                css={{ marginLeft: 4, marginRight: 0 }}
              >
                +{normalizedGroupBy.groupByKeys.length - 1}
              </Tag>
            )}
            {groupBy && (
              <XCircleFillIcon
                aria-hidden="false"
                css={{
                  color: theme.colors.textPlaceholder,
                  fontSize: theme.typography.fontSizeSm,
                  marginLeft: theme.spacing.sm,

                  ':hover': {
                    color: theme.colors.actionTertiaryTextHover,
                  },
                }}
                role="button"
                onClick={() => {
                  onChange(null);
                }}
                onPointerDownCapture={(e) => {
                  // Prevents the dropdown from opening when clearing
                  e.stopPropagation();
                }}
              />
            )}
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content>
          {isLoading ? (
            <DropdownMenu.Item componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsgroupbyselector.tsx_436">
              <Spinner />
            </DropdownMenu.Item>
          ) : (
            <GroupBySelectorBody
              groupBy={normalizedGroupBy}
              onChange={onChange}
              runsData={runsData}
              onUseGroupedValuesInChartsChange={onUseGroupedValuesInChartsChange}
              useGroupedValuesInCharts={useGroupedValuesInCharts}
            />
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    );
  },
);

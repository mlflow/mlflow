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
  Tooltip,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { compact, isEmpty, keys, uniq, values } from 'lodash';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MLFLOW_INTERNAL_PREFIX } from '../../../../../common/utils/TagUtils';
import { createRunsGroupByKey, parseRunsGroupByKey } from '../../utils/experimentPage.group-row-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { RunGroupingAggregateFunction, RunGroupingMode } from '../../utils/experimentPage.row-types';

export interface ExperimentViewRunsGroupBySelectorProps {
  runsData: ExperimentRunsSelectorResult;
  groupBy: string;
  onChange: (newGroupByKey: string) => void;
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
}: {
  groupBy: string;
  onChange: (newGroupByKey: string) => void;
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
  const {
    aggregateFunction = RunGroupingAggregateFunction.Average,
    groupByData,
    mode,
  } = parseRunsGroupByKey(groupBy) || {};

  const currentAggregateFunctionLabel = {
    min: minimumLabel,
    max: maximumLabel,
    average: averageLabel,
  }[aggregateFunction];

  const groupByValue = groupByData || '';
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

  return (
    <>
      <div css={{ display: 'flex', gap: theme.spacing.xs, padding: theme.spacing.sm }}>
        <Input
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
            e.stopPropagation();
          }}
        />
        <DropdownMenu.Root>
          <Tooltip
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
          </Tooltip>
          <DropdownMenu.Content align="start" side="right">
            <DropdownMenu.RadioGroup value={groupBy} onValueChange={onChange}>
              <DropdownMenu.RadioItem
                disabled={!groupByValue}
                value={createRunsGroupByKey(mode, groupByValue, RunGroupingAggregateFunction.Min) || minimumLabel}
                key={createRunsGroupByKey(mode, groupByValue, RunGroupingAggregateFunction.Min) || minimumLabel}
              >
                <DropdownMenu.ItemIndicator />
                {minimumLabel}
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem
                disabled={!groupByValue}
                value={createRunsGroupByKey(mode, groupByValue, RunGroupingAggregateFunction.Max) || maximumLabel}
                key={createRunsGroupByKey(mode, groupByValue, RunGroupingAggregateFunction.Max) || maximumLabel}
              >
                <DropdownMenu.ItemIndicator />
                {maximumLabel}
              </DropdownMenu.RadioItem>
              <DropdownMenu.RadioItem
                disabled={!groupByValue}
                value={createRunsGroupByKey(mode, groupByValue, RunGroupingAggregateFunction.Average) || averageLabel}
                key={createRunsGroupByKey(mode, groupByValue, RunGroupingAggregateFunction.Average) || averageLabel}
              >
                <DropdownMenu.ItemIndicator />
                {averageLabel}
              </DropdownMenu.RadioItem>
            </DropdownMenu.RadioGroup>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
      <DropdownMenu.Group css={{ maxHeight: 400, overflowY: 'scroll' }}>
        <DropdownMenu.RadioGroup value={groupBy} onValueChange={onChange}>
          {attributesMatchFilter && (
            <>
              <DropdownMenu.Label>
                <FormattedMessage {...messages.attributes} />
              </DropdownMenu.Label>
              {datasetLabel.toLowerCase().includes(filter.toLowerCase()) && (
                <DropdownMenu.RadioItem
                  value={createRunsGroupByKey(RunGroupingMode.Dataset, 'dataset', aggregateFunction)}
                  key={createRunsGroupByKey(RunGroupingMode.Dataset, 'dataset', aggregateFunction)}
                  ref={attributeElementRef}
                >
                  <DropdownMenu.ItemIndicator />
                  {datasetLabel}
                </DropdownMenu.RadioItem>
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
                  <DropdownMenu.RadioItem
                    value={groupByKey}
                    key={groupByKey}
                    ref={index === 0 ? tagElementRef : undefined}
                  >
                    <DropdownMenu.ItemIndicator />
                    {tagName}
                  </DropdownMenu.RadioItem>
                );
              })}
              {!tagNames.length && (
                <DropdownMenu.Item disabled>
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
                  <DropdownMenu.RadioItem
                    value={groupByKey}
                    key={groupByKey}
                    ref={index === 0 ? paramElementRef : undefined}
                  >
                    <DropdownMenu.ItemIndicator />
                    {paramName}
                  </DropdownMenu.RadioItem>
                );
              })}
              {!runsData.paramKeyList.length && (
                <DropdownMenu.Item disabled>
                  <FormattedMessage {...messages.noParams} />
                </DropdownMenu.Item>
              )}
            </>
          )}
          {!hasAnyResults && (
            <DropdownMenu.Item disabled>
              <FormattedMessage {...messages.noResults} />
            </DropdownMenu.Item>
          )}
        </DropdownMenu.RadioGroup>
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
    onChange,
    isLoading,
  }: ExperimentViewRunsGroupBySelectorProps & {
    groupBy: string;
    onChange: (newGroupByKey: string) => void;
    isLoading: boolean;
  }) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();
    const { mode, groupByData } = parseRunsGroupByKey(groupBy) || {};

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
            {groupByData ? (
              <FormattedMessage
                defaultMessage="Group: {value}"
                description="Experiment page > group by runs control > trigger button label > with value"
                values={{
                  value: mode === RunGroupingMode.Dataset ? intl.formatMessage(messages.dataset) : groupByData,
                }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Group by"
                description="Experiment page > group by runs control > trigger button label > empty"
              />
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
                  onChange('');
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
            <DropdownMenu.Item>
              <Spinner />
            </DropdownMenu.Item>
          ) : (
            <GroupBySelectorBody groupBy={groupBy} onChange={onChange} runsData={runsData} />
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    );
  },
);

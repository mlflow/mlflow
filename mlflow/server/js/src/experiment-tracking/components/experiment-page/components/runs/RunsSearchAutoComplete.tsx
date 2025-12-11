import { useMemo, useRef } from 'react';
import type { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { EntitySearchAutoComplete } from '../../../EntitySearchAutoComplete';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { RunsSearchTooltipContent } from './RunsSearchTooltipContent';
import type {
  EntitySearchAutoCompleteEntityNameGroup,
  EntitySearchAutoCompleteOptionGroup,
} from '../../../EntitySearchAutoComplete.utils';
import {
  cleanEntitySearchTagNames,
  getEntitySearchOptionsFromEntityNames,
} from '../../../EntitySearchAutoComplete.utils';
import { shouldUseRegexpBasedAutoRunsSearchFilter } from '../../../../../common/utils/FeatureUtils';

// A default placeholder for the search box
const SEARCH_BOX_PLACEHOLDER = 'metrics.rmse < 1 and params.model = "tree"';

export type RunsSearchAutoCompleteProps = {
  runsData: ExperimentRunsSelectorResult;
  searchFilter: string;
  onSearchFilterChange: (newValue: string) => void;
  onClear: () => void;
  requestError: ErrorWrapper | Error | null;
  className?: string;
};

const ATTRIBUTE_OPTIONS = [
  'run_id',
  'run_name',
  'status',
  'artifact_uri',
  'user_id',
  'start_time',
  'end_time',
  'created',
].map((s) => ({ value: `attributes.${s}` }));

const mergeDeduplicate = (list1: string[], list2: string[]) => [...new Set([...list1, ...list2])];
const getTagNames = (tagsList: any[]) => tagsList.flatMap((tagRecord) => Object.keys(tagRecord));

const getEntityNamesFromRunsData = (
  newRunsData: ExperimentRunsSelectorResult,
  existingNames: EntitySearchAutoCompleteEntityNameGroup,
): EntitySearchAutoCompleteEntityNameGroup => {
  const metricNames = mergeDeduplicate(existingNames.metricNames, newRunsData.metricKeyList);
  const paramNames = mergeDeduplicate(existingNames.paramNames, newRunsData.paramKeyList);
  const tagNames = cleanEntitySearchTagNames(
    mergeDeduplicate(getTagNames(existingNames.tagNames), getTagNames(newRunsData.tagsList)),
  );

  return {
    metricNames,
    paramNames,
    tagNames,
  };
};

export const RunsSearchAutoComplete = ({ runsData, ...restProps }: RunsSearchAutoCompleteProps) => {
  const existingEntityNamesRef = useRef<EntitySearchAutoCompleteEntityNameGroup>({
    metricNames: [],
    paramNames: [],
    tagNames: [],
  });

  const baseOptions = useMemo<EntitySearchAutoCompleteOptionGroup[]>(() => {
    const existingEntityNames = existingEntityNamesRef.current;
    const mergedEntityNames = getEntityNamesFromRunsData(runsData, existingEntityNames);
    existingEntityNamesRef.current = mergedEntityNames;
    return getEntitySearchOptionsFromEntityNames(mergedEntityNames, ATTRIBUTE_OPTIONS);
  }, [runsData]);

  return (
    <EntitySearchAutoComplete
      {...restProps}
      baseOptions={baseOptions}
      tooltipContent={<RunsSearchTooltipContent />}
      placeholder={SEARCH_BOX_PLACEHOLDER}
      useQuickFilter={shouldUseRegexpBasedAutoRunsSearchFilter()}
    />
  );
};

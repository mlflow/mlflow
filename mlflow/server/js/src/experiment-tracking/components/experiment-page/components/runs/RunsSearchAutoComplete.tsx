import { useMemo, useRef } from 'react';
import { useIntl } from 'react-intl';
import type { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { EntitySearchAutoComplete } from '../../../EntitySearchAutoComplete';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { RunsSearchTooltipContent } from './RunsSearchTooltipContent';
import type {
  EntitySearchAutoCompleteEntityNameGroup,
  EntitySearchAutoCompleteOptionGroup,
} from '../../../EntitySearchAutoComplete.utils';
import { cleanEntitySearchTagNames } from '../../../EntitySearchAutoComplete.utils';
import { shouldUseRegexpBasedAutoRunsSearchFilter } from '../../../../../common/utils/FeatureUtils';
import { GIT_SOURCE_TAGS, isGitSourceTag } from '../../../../utils/gitSourceTags';

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
  // Keep raw tag names in the cache. Cleaning (filter + backtick-wrap) happens only at output
  // time so that on re-render the cache doesn't get re-cleaned, which would double-wrap names
  // and accumulate spurious entries.
  const tagNames = mergeDeduplicate(existingNames.tagNames, getTagNames(newRunsData.tagsList));

  return {
    metricNames,
    paramNames,
    tagNames,
  };
};

export const RunsSearchAutoComplete = ({ runsData, ...restProps }: RunsSearchAutoCompleteProps) => {
  const intl = useIntl();
  const existingEntityNamesRef = useRef<EntitySearchAutoCompleteEntityNameGroup>({
    metricNames: [],
    paramNames: [],
    tagNames: [],
  });

  const baseOptions = useMemo<EntitySearchAutoCompleteOptionGroup[]>(() => {
    const existingEntityNames = existingEntityNamesRef.current;
    const mergedEntityNames = getEntityNamesFromRunsData(runsData, existingEntityNames);
    existingEntityNamesRef.current = mergedEntityNames;

    // Split git source tags out of the generic Tags section so they can be surfaced in a
    // dedicated Git section with friendly labels (e.g. "commit" instead of
    // `mlflow.source.git.commit`).
    const gitTagNames = mergedEntityNames.tagNames.filter(isGitSourceTag);
    const nonGitTagNames = mergedEntityNames.tagNames.filter((tag) => !isGitSourceTag(tag));

    return [
      {
        label: 'Metrics',
        options: mergedEntityNames.metricNames.map((m) => ({ value: `metrics.${m}` })),
      },
      {
        label: 'Parameters',
        options: mergedEntityNames.paramNames.map((p) => ({ value: `params.${p}` })),
      },
      {
        label: 'Tags',
        options: cleanEntitySearchTagNames(nonGitTagNames).map((t) => ({ value: `tags.${t}` })),
      },
      {
        label: 'Git',
        options: gitTagNames.map((tag) => {
          const friendlyLabel = intl.formatMessage(GIT_SOURCE_TAGS[tag].short);
          return {
            value: `tags.\`${tag}\``,
            label: friendlyLabel,
            // Make the friendly label searchable too, so typing "repository" finds repoURL etc.
            searchText: `${tag} ${friendlyLabel}`,
          };
        }),
      },
      {
        label: 'Attributes',
        options: ATTRIBUTE_OPTIONS,
      },
    ];
  }, [runsData, intl]);

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

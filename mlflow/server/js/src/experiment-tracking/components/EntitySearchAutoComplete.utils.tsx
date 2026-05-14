import React from 'react';
import { shouldEnableMinMaxMetricsOnExperimentPage } from '../../common/utils/FeatureUtils';
import { MLFLOW_INTERNAL_PREFIX } from '../../common/utils/TagUtils';
import { sanitizeStringForRegexp } from '../../common/utils/StringUtils';

export type EntitySearchAutoCompleteOption = {
  label?: string | React.ReactNode;
  value: string;
  /**
   * Optional extra text to match against when filtering. Use for options whose user-visible label
   * differs from the underlying `value` and where the label should also be searchable (e.g. the
   * friendly "commit" / "branch" labels for git source tags whose values are technical paths).
   */
  searchText?: string;
};

export type EntitySearchAutoCompleteOptionGroup = {
  label: string;
  options: EntitySearchAutoCompleteOption[];
};

export type EntitySearchAutoCompleteEntity = {
  name: string;
  startIndex: number;
  endIndex: number;
};

type EntitySearchAutoCompleteClause = {
  clause: string;
  startIndex: number;
};

export type EntitySearchAutoCompleteEntityNameGroup = {
  metricNames: string[];
  paramNames: string[];
  tagNames: string[];
};

/**
 * Given an input string, returns a list of Clause objects
 * containing the clauses in the input and the indices of their
 * starting positions in the overall string.
 */
const getClausesAndStartIndex = (str: string) => {
  const re = /and[\s]+/gi;
  const results: EntitySearchAutoCompleteClause[] = [];
  let match, position;
  while (((position = re.lastIndex), (match = re.exec(str)))) {
    results.push({ clause: str.substring(position, match.index), startIndex: position });
  }
  results.push({ clause: str.substring(position), startIndex: position });
  return results;
};

/**
 * Internal mlflow.* tags that are useful enough as filter dimensions to surface in autocomplete
 * alongside user-defined tags. Keep this list narrow.
 */
const SEARCHABLE_INTERNAL_TAGS = new Set<string>([
  'mlflow.source.git.commit',
  'mlflow.source.git.branch',
  'mlflow.source.git.repoURL',
]);

/**
 * Filters out internal tag names (except for a small allowlist) and wraps names that include
 * control characters in backticks.
 */
export const cleanEntitySearchTagNames = (tagNames: string[]) =>
  tagNames
    .filter((tag: string) => !tag.startsWith(MLFLOW_INTERNAL_PREFIX) || SEARCHABLE_INTERNAL_TAGS.has(tag))
    .map((tag: string) => {
      if (tag.includes('"') || tag.includes(' ') || tag.includes('.')) {
        return `\`${tag}\``;
      } else if (tag.includes('`')) {
        return `"${tag}"`;
      } else return tag;
    });

export const getEntitySearchOptionsFromEntityNames = (
  entityNames: EntitySearchAutoCompleteEntityNameGroup,
  attributeOptions: EntitySearchAutoCompleteOption[],
): EntitySearchAutoCompleteOptionGroup[] => [
  {
    label: 'Metrics',
    options: entityNames.metricNames.map((m) => ({ value: `metrics.${m}` })),
  },
  {
    label: 'Parameters',
    options: entityNames.paramNames.map((p) => ({ value: `params.${p}` })),
  },
  {
    label: 'Tags',
    options: entityNames.tagNames.map((t) => ({ value: `tags.${t}` })),
  },
  {
    label: 'Attributes',
    options: attributeOptions,
  },
];

// Bolds a specified segment of `wholeText`.
const boldedText = (wholeText: string, shouldBeBold: string) => {
  // Escape all the special characters
  const escapedText = sanitizeStringForRegexp(shouldBeBold);

  const textArray = wholeText.split(RegExp(escapedText, 'ig'));
  const match = wholeText.match(RegExp(escapedText, 'ig'));

  return (
    // Autocomplete sets font weight to 600 on full match resulting in double bolding.
    // Override this here
    <span css={{ fontWeight: 'normal' }} data-testid={wholeText}>
      {textArray.map((item, index) => (
        <React.Fragment key={index}>
          {item}
          {index !== textArray.length - 1 && match && <b>{match[index]}</b>}
        </React.Fragment>
      ))}
    </span>
  );
};

/**
 * Given an input string, returns a list of Entity objects
 * containing the search entities in the input and their
 * start and end indices in the whole string.
 */
export const getEntitySearchEntitiesAndIndices = (str: string) => {
  const re = />|<|>=|<=|=|!=|like|ilike/gi;
  const clauses = getClausesAndStartIndex(str);
  const results: EntitySearchAutoCompleteEntity[] = [];
  clauses.forEach((clauseObj) => {
    const clauseText = clauseObj.clause;
    const entity = clauseText.split(re)[0];
    const { startIndex } = clauseObj;
    results.push({
      name: entity,
      startIndex: 0 + startIndex,
      endIndex: entity.length + startIndex,
    });
  });
  return results;
};

export const getFilteredOptionsFromEntityName = (
  baseOptions: EntitySearchAutoCompleteOptionGroup[],
  entityBeingEdited: EntitySearchAutoCompleteEntity,
  suggestionLimits: Record<string, number>,
): EntitySearchAutoCompleteOptionGroup[] => {
  const needle = entityBeingEdited.name.toLowerCase().trim();
  return baseOptions
    .map((group) => {
      const newOptions = group.options
        .filter(
          (option) =>
            option.value.toLowerCase().includes(needle) ||
            (option.searchText !== undefined && option.searchText.toLowerCase().includes(needle)),
        )
        .map((match) => ({
          value: match.value,
          // Preserve an explicit label if one was set on the option (e.g. a friendly label for
          // git source tags). Otherwise fall back to the auto-bolded value.
          label: match.label ?? boldedText(match.value, entityBeingEdited.name.trim()),
        }));
      const limitForGroup = suggestionLimits[group.label];
      const ellipsized = [
        ...newOptions.slice(0, limitForGroup),
        ...(newOptions.length > limitForGroup ? [{ label: '...', value: `..._${group.label}` }] : []),
      ];
      return {
        label: group.label,
        options: ellipsized,
      };
    })
    .filter((group) => group.options.length > 0);
};

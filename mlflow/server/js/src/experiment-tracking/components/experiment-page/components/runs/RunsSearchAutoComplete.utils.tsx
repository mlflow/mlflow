import React from 'react';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';

export type Option = {
  label?: string | React.ReactNode;
  value: string;
};

export type OptionGroup = {
  label: string;
  options: Option[];
};

export type Entity = {
  name: string;
  startIndex: number;
  endIndex: number;
};

export type Clause = {
  clause: string;
  startIndex: number;
};

export type EntityNameGroup = {
  metricNames: string[];
  paramNames: string[];
  tagNames: string[];
};

export const ATTRIBUTE_OPTIONS = [
  'run_id',
  'run_name',
  'status',
  'artifact_uri',
  'user_id',
  'start_time',
  'end_time',
  'created',
].map((s) => ({ value: `attributes.${s}` }));

export const getEntityNamesFromRunsData = (
  newRunsData: ExperimentRunsSelectorResult,
  existingNames: EntityNameGroup,
): EntityNameGroup => {
  const mergeDedup = (list1: any[], list2: any[]) => [...new Set([...list1, ...list2])];
  const getTagNames = (tagsList: any[]) => tagsList.flatMap((tagRecord) => Object.keys(tagRecord));

  const metricNames = mergeDedup(existingNames.metricNames, newRunsData.metricKeyList);
  const paramNames = mergeDedup(existingNames.paramNames, newRunsData.paramKeyList);
  const tagNames = mergeDedup(getTagNames(existingNames.tagNames), getTagNames(newRunsData.tagsList));
  // Filter out internal tag names and wrap names that include control characters in backticks.
  const tagNamesCleaned = tagNames
    .filter((s: string) => !s.startsWith('mlflow.'))
    .map((s: string) => {
      if (s.includes('"') || s.includes(' ') || s.includes('.')) {
        return `\`${s}\``;
      } else if (s.includes('`')) {
        return `"${s}"`;
      } else return s;
    });
  return {
    metricNames,
    paramNames,
    tagNames: tagNamesCleaned,
  };
};

export const getOptionsFromEntityNames = (entityNames: EntityNameGroup): OptionGroup[] => [
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
    options: ATTRIBUTE_OPTIONS,
  },
];

// Bolds a specified segment of `wholeText`.
const boldedText = (wholeText: string, shouldBeBold: string) => {
  const textArray = wholeText.split(RegExp(shouldBeBold.replace('.', '\\.'), 'ig'));
  const match = wholeText.match(RegExp(shouldBeBold.replace('.', '\\.'), 'ig'));

  return (
    // Autocomplete sets font weight to 600 on full match resulting in double bolding.
    // Override this here
    <span css={{ fontWeight: 'normal' }} data-test-id={wholeText}>
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
 * Given an input string, returns a list of Clause objects
 * containing the clauses in the input and the indices of their
 * starting positions in the overall string.
 */
const getClausesAndStartIndex = (str: string) => {
  const re = /and[\s]+/gi;
  const results: Clause[] = [];
  let match, position;
  while (((position = re.lastIndex), (match = re.exec(str)))) {
    results.push({ clause: str.substring(position, match.index), startIndex: position });
  }
  results.push({ clause: str.substring(position), startIndex: position });
  return results;
};

/**
 * Given an input string, returns a list of Entity objects
 * containing the search entities in the input and their
 * start and end indices in the whole string.
 */

export const getEntitiesAndIndices = (str: string) => {
  const re = />|<|>=|<=|=|!=|like|ilike/gi;
  const clauses = getClausesAndStartIndex(str);
  const results: Entity[] = [];
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
  baseOptions: OptionGroup[],
  entityBeingEdited: Entity,
  suggestionLimits: any,
): OptionGroup[] => {
  return baseOptions
    .map((group) => {
      const newOptions = group.options
        .filter((option) => option.value.toLowerCase().includes(entityBeingEdited.name.toLowerCase().trim()))
        .map((match) => ({
          value: match.value,
          label: boldedText(match.value, entityBeingEdited.name.trim()),
        }));
      const limitForGroup = (suggestionLimits as any)[group.label];
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

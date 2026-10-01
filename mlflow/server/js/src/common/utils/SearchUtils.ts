// Avoid treating operator words in ordinary phrases as structured search syntax.
const FILTER_IDENTIFIER_PATTERN = String.raw`[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|"[^"]+"|\x60[^\x60]+\x60))*`;
const FILTER_CLAUSE_PATTERN = String.raw`${FILTER_IDENTIFIER_PATTERN}(?:\s+(?:(?:ILIKE|LIKE)\s+(?:"[^"]*"|'[^']*')|(?:NOT\s+)?IN\s+\(\s*(?:"[^"]*"|'[^']*')(?:\s*,\s*(?:"[^"]*"|'[^']*'))*\s*\)|IS\s+(?:NOT\s+)?NULL)|\s*(?:=|!=|<=|>=|<|>)\s*(?:"[^"]*"|'[^']*'|-?\d+(?:\.\d+)?|TRUE|FALSE|NULL))`;
const SQL_FILTER_PATTERN = new RegExp(
  `^\\s*${FILTER_CLAUSE_PATTERN}(?:\\s+AND\\s+${FILTER_CLAUSE_PATTERN})*\\s*$`,
  'i',
);

type SearchParamValue = string | number | string[] | undefined;

export const buildSearchParams = (params: Record<string, SearchParamValue>): string => {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) {
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        searchParams.append(key, item);
      }
    } else {
      searchParams.append(key, String(value));
    }
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
};

/**
 * Builds a filter clause from a search string.
 * If the input is a valid filter expression, it is passed through as-is.
 * Otherwise, it is treated as a plain name search with special SQL characters
 * escaped.
 */
export const buildSearchFilterClause = (searchFilter?: string, fieldName = 'name'): string | undefined => {
  if (!searchFilter) {
    return undefined;
  }

  if (SQL_FILTER_PATTERN.test(searchFilter)) {
    return searchFilter;
  }

  const escaped = searchFilter.replace(/'/g, "''").replace(/%/g, '\\%').replace(/_/g, '\\_');
  return `${fieldName} ILIKE '%${escaped}%'`;
};

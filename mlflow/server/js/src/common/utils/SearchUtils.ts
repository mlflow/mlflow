// Avoid treating operator words in ordinary phrases as structured search syntax.
const UNQUOTED_IDENTIFIER_PATTERN = String.raw`[A-Za-z_][A-Za-z0-9_]*`;
const QUOTED_IDENTIFIER_PATTERN = String.raw`(?:"[^"]+"|\x60[^\x60]+\x60)`;
const IDENTIFIER_SUFFIX_PATTERN = String.raw`(?:\.(?:${UNQUOTED_IDENTIFIER_PATTERN}|${QUOTED_IDENTIFIER_PATTERN}))*`;

const createSqlFilterPattern = (allowQuotedFirstSegment: boolean) => {
  const firstSegment = allowQuotedFirstSegment
    ? String.raw`(?:${UNQUOTED_IDENTIFIER_PATTERN}|${QUOTED_IDENTIFIER_PATTERN})`
    : UNQUOTED_IDENTIFIER_PATTERN;
  const identifier = `${firstSegment}${IDENTIFIER_SUFFIX_PATTERN}`;
  const clause = String.raw`${identifier}(?:\s+(?:(?:ILIKE|LIKE)\s+(?:"[^"]*"|'[^']*')|(?:NOT\s+)?IN\s+\(\s*(?:"[^"]*"|'[^']*')(?:\s*,\s*(?:"[^"]*"|'[^']*'))*\s*\)|IS\s+(?:NOT\s+)?NULL)|\s*(?:=|!=|<=|>=|<|>)\s*(?:"[^"]*"|'[^']*'|-?\d+(?:\.\d+)?|TRUE|FALSE|NULL))`;
  return new RegExp(`^\\s*${clause}(?:\\s+AND\\s+${clause})*\\s*$`, 'i');
};

const SQL_FILTER_PATTERN = createSqlFilterPattern(false);
const SQL_FILTER_WITH_QUOTED_FIRST_SEGMENT_PATTERN = createSqlFilterPattern(true);

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
 * Enable quoted first segments only for search backends that accept them.
 */
export const buildSearchFilterClause = (
  searchFilter?: string,
  fieldName = 'name',
  { allowQuotedFirstSegment = false }: { allowQuotedFirstSegment?: boolean } = {},
): string | undefined => {
  if (!searchFilter) {
    return undefined;
  }

  const filterPattern = allowQuotedFirstSegment ? SQL_FILTER_WITH_QUOTED_FIRST_SEGMENT_PATTERN : SQL_FILTER_PATTERN;
  if (filterPattern.test(searchFilter)) {
    return searchFilter;
  }

  const escaped = searchFilter.replace(/'/g, "''").replace(/%/g, '\\%').replace(/_/g, '\\_');
  return `${fieldName} ILIKE '%${escaped}%'`;
};

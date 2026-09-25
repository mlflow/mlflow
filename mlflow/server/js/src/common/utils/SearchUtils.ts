const SQL_KEYWORD_PATTERN = /(\s+(ILIKE|LIKE|IN|IS)\s+)|=|!=|<=|>=|<|>/i;

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
 * If the input contains SQL-like operators (ILIKE, LIKE, IN, IS, =, !=, etc.),
 * it is passed through as-is. Otherwise, it is treated as a plain name search
 * with special SQL characters escaped.
 */
export const buildSearchFilterClause = (searchFilter?: string, fieldName = 'name'): string | undefined => {
  if (!searchFilter) {
    return undefined;
  }

  if (SQL_KEYWORD_PATTERN.test(searchFilter)) {
    return searchFilter;
  }

  const escaped = searchFilter.replace(/'/g, "''").replace(/%/g, '\\%').replace(/_/g, '\\_');
  return `${fieldName} ILIKE '%${escaped}%'`;
};

export class SearchUtils {
  static validateSearchInput(searchInput) {
    const trimmedInput = searchInput.trim();
    if (trimmedInput === '') {
      return;
    }
    const searchClauses = searchInput.split("and");
    searchClauses.forEach((clause) => Private.validateSearchClause(clause));
  }
}

const METRIC_CLAUSE_REGEX = /metrics\.([a-zA-z0-9.]+)\s{0,}(=|!=|>|>=|<=|<)\s{0,}(\d+\.{0,}\d{0,}|\.+\d+)/;
const PARAM_CLAUSE_REGEX = /params\.([a-zA-z0-9.]+)\s{0,}(=|!=)\s{0,}"(.*)"/;
const TAG_CLAUSE_REGEX = /tags\.([a-zA-z0-9.]+)\s{0,}(=|!=)\s{0,}"(.*)"/;

class Private {
  static validateSearchClause(searchClauseString) {
    const trimmedInput = searchClauseString.trim();
    const metricMatches = METRIC_CLAUSE_REGEX.exec(trimmedInput);
    const paramMatches = PARAM_CLAUSE_REGEX.exec(trimmedInput);
    const tagMatches = TAG_CLAUSE_REGEX.exec(trimmedInput);
    if (!metricMatches && !paramMatches && !tagMatches) {
      throw new SearchError("The search input should be like 'metrics.alpha >= 0.9', " +
        "'params.file = \"test.txt\"', or 'tags.mlflow.parentRunId = \"abc\"'.");
    }
  }
}

export class SearchError extends Error {
  constructor(...params) {
    super(...params);
    // Code copied from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
    // Global_Objects/Error#Custom_Error_Types
    // Maintains proper stack trace for where our error was thrown (only available on V8)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, SearchError);
    }
  }
}


export class SearchUtils {
  static parseSearchInput(searchInput) {
    const trimmedInput = searchInput.trim();
    if (trimmedInput === '') {
      return [];
    }
    const searchClauses = searchInput.split("and");
    return searchClauses.map((clause) => Private.parseSearchClause(clause));
  }
}

const METRIC_CLAUSE_REGEX = /metrics\.([a-zA-z0-9]+)\s{0,}(=|!=|>|>=|<=|<)\s{0,}(\d+\.{0,}\d{0,})/;
const PARAM_CLAUSE_REGEX = /params\.([a-zA-z0-9]+)\s{0,}(=|!=)\s{0,}"(.*)"/;
class Private {
  static parseSearchClause(searchClauseString) {
    const trimmedInput = searchClauseString.trim();
    const metricMatches = METRIC_CLAUSE_REGEX.exec(trimmedInput);
    if (metricMatches) {
      return {
        metric: {
          key: metricMatches[1],
          double: {
            comparator: metricMatches[2],
            value: parseFloat(metricMatches[3]),
          }
        }
      };
    }
    const paramMatches = PARAM_CLAUSE_REGEX.exec(trimmedInput);
    if (paramMatches) {
      return {
        parameter: {
          key: paramMatches[1],
          string: {
            comparator: paramMatches[2],
            value: paramMatches[3],
          }
        }
      };
    }
    throw new SearchError("The search input should be like 'metrics.alpha >= 0.9' or " +
     "'params.file = \"test.txt\"'.");
  }
}

export class SearchError {
  constructor(errorMessage) {
    this.errorMessage = errorMessage;
  }
}


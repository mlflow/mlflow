import { REGISTERED_MODELS_SEARCH_NAME_FIELD } from '../constants';
import { resolveFilterValue } from '../actions';

export function getModelNameFilter(query: string): string {
  if (query) {
    return `${REGISTERED_MODELS_SEARCH_NAME_FIELD} ilike ${resolveFilterValue(query, true)}`;
  } else {
    return '';
  }
}

export function getCombinedSearchFilter({
  query = '',
}: {
  query?: string;
} = {}) {
  const filters = [];
  const initialFilter = query.includes('tags.') ? query : getModelNameFilter(query);
  if (initialFilter) filters.push(initialFilter);
  return filters.join(' AND ');
}

export function constructSearchInputFromURLState(urlState: Record<string, string>): string {
  if ('searchInput' in urlState) {
    return urlState['searchInput'];
  }
  if ('nameSearchInput' in urlState && 'tagSearchInput' in urlState) {
    return getModelNameFilter(urlState['nameSearchInput']) + ` AND ` + urlState['tagSearchInput'];
  }
  if ('tagSearchInput' in urlState) {
    return urlState['tagSearchInput'];
  }
  if ('nameSearchInput' in urlState) {
    return urlState['nameSearchInput'];
  }
  return '';
}

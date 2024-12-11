/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { REGISTERED_MODELS_SEARCH_NAME_FIELD } from '../constants';
import { resolveFilterValue } from '../actions';

export function getModelNameFilter(query: any) {
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

export function constructSearchInputFromURLState(urlState: any) {
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

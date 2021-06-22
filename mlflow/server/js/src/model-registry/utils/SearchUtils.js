import { REGISTERED_MODELS_SEARCH_NAME_FIELD } from '../constants';
import { resolveFilterValue } from '../actions';

export function getModelNameFilter(query) {
  return `${REGISTERED_MODELS_SEARCH_NAME_FIELD} ilike ${resolveFilterValue(query, true)}`;
}

export function appendTagsFilter(nameQuery, tagsQuery) {
  if (!nameQuery && !tagsQuery) {
    return '';
  }
  if (!nameQuery) {
    return tagsQuery;
  }
  if (!tagsQuery) {
    return nameQuery;
  }
  return nameQuery + ` AND ` + tagsQuery;
}

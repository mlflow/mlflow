import { getJson } from '../utils/FetchUtils';

export class UnityCatalogService {
  /**
   * List all UC catalogs.
   */
  static listCatalogs = () => getJson({ relativeUrl: 'ajax-api/2.0/unity-catalog/catalogs' });
}

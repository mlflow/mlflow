import { getJson } from '../utils/FetchUtils';

export class UnityCatalogService {
  /**
   * List all UC catalogs.
   */
  static listCatalogs = () => getJson({ relativeUrl: 'ajax-api/2.0/unity-catalog/catalogs' });

  /**
   * Get a UC table.
   */
  static getTable = (data) => {
    const { tableName, ...otherData } = data;

    return getJson({
      ...otherData,
      relativeUrl: `ajax-api/2.0/unity-catalog/tables/${encodeURIComponent(tableName)}`,
    });
  };
}

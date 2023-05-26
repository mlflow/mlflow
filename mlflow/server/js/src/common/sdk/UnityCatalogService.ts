/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { getJson } from '../utils/FetchUtils';

export class UnityCatalogService {
  /**
   * List all UC catalogs.
   */
  static listCatalogs = () => getJson({ relativeUrl: 'ajax-api/2.0/unity-catalog/catalogs' });

  /**
   * Get a UC table.
   */
  static getTable = (data: any) => {
    const { tableName, ...otherData } = data;

    return getJson({
      ...otherData,
      relativeUrl: `ajax-api/2.0/unity-catalog/tables/${encodeURIComponent(tableName)}`,
    });
  };
}

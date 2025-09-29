/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class ModelBuilder {
  /**
   * @param {Immutable.Record schema} AnemicRecord: generated Immutable Record
   * @param {Object<string, func>} prototypeFuncs: functions to add to
   *   AnemicRecord's prototype
   */
  static extend(AnemicRecord: any, prototypeFuncs: any) {
    const FatRecord = class FatRecord extends AnemicRecord {};
    Object.keys(prototypeFuncs).forEach((funcName) => {
      if (FatRecord.prototype[funcName]) {
        throw new Error(`Duplicate prototype function: ${funcName} already exists on the model.`);
      }
      FatRecord.prototype[funcName] = prototypeFuncs[funcName];
    });

    return FatRecord;
  }
}

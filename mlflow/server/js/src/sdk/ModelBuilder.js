export class ModelBuilder {
  /**
   * @param {Immutable.Record schema} AnemicRecord: generated Immutable Record
   * @param {Object<string, func>} prototypeFuncs: functions to add to
   *   AnemicRecord's prototype
   */
  static extend(AnemicRecord, prototypeFuncs) {
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

/**
 * This function constructs protobuf-generated immutable.js entity instance by
 * consuming compatible data in the form of POJO. It accepts the Entity
 * schema object as either `fromJs`-enhanced entity (or just a constructor) and
 * returns the factory function that consumes the POJO.
 *
 * `R` type template parameter indicates the plain data model
 * `T` type template parameter is the protobuf-generated entity class that contains `fromJs` in the prototype chain
 *
 * @example
 *
 * const Sample = Immutable.Record({
 *   field: undefined,
 * }, 'Sample');
 *
 * const immutableEntity = hydrateRecord(Sample)({
 *   field: 1234,
 * }
 */
export const hydrateImmutableRecord =
  <R, T extends { fromJs?: (data: R) => any }>(classType: T | any) =>
  (data: R) =>
    classType.fromJs?.(data) || classType(data);

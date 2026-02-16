/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class RecordUtils {
  /**
   * This function is our implementation of Immutable.fromJS, but fixes a bug with
   * their implementation.
   * @param {object} pojo - a javascript object that we want to convert to an
   *   Immutable Record
   * @param {function} fromJsReviver - a function that takes a key and value
   *   and returns the value as an Immutable object, based on the key
   * @return {object} still a plain javascript object, but all the nested objects
   *   have already been converted to Immutable types so you can do:
   *     new RecordType(RecordUtils.fromJs({...}, RecordType.fromJsReviver));
   *   to create an Immutable Record with nested Immutables.
   */
  static fromJs(pojo: any, fromJsReviver: any) {
    const record = {};
    for (const key in pojo) {
      // don't convert keys with undefined value
      if (pojo.hasOwnProperty(key) && pojo[key] !== undefined) {
        // Record an event when the value is null, since if it's null and we still create the
        // object, it might cause some bug CJ-18735
        if (pojo[key] === null) {
          (window as any).recordEvent(
            'clientsideEvent',
            {
              eventType: 'nullValueForNestedProto',
              property: key,
            },
            pojo,
          );
        }
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        record[key] = fromJsReviver(key, pojo[key]);
      }
    }
    return record;
  }
}

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
  static fromJs(pojo, fromJsReviver) {
    const record = {};
    for (const key in pojo) {
      // don't convert keys with undefined value
      if (pojo.hasOwnProperty(key) && pojo[key] !== undefined) {
        // Record an event when the value is null, since if it's null and we still create the
        // object, it might cause some bug CJ-18735
        if (pojo[key] === null) {
          window.recordEvent(
            'clientsideEvent',
            {
              eventType: 'nullValueForNestedProto',
              property: key,
            },
            pojo,
          );
        }
        record[key] = fromJsReviver(key, pojo[key]);
      }
    }
    return record;
  }
}

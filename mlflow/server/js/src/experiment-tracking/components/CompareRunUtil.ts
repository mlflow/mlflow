export default class CompareRunUtil {
  /**
   * Find in a list of metrics/params a certain key
   */
  static findInList(data: any, key: any) {
    let found = undefined;
    data.forEach((value: any) => {
      if (value.key === key) {
        found = value;
      }
    });
    return found;
  }

  /**
   * Get all keys present in the data in ParamLists or MetricLists or Schema input and outputs lists
   */
  static getKeys(lists: any, numeric: any) {
    const keys = {};
    lists.forEach((list: any) =>
      list.forEach((item: any) => {
        if (!(item.key in keys)) {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          keys[item.key] = true;
        }
        if (numeric && isNaN(parseFloat(item.value))) {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          keys[item.key] = false;
        }
      }),
    );
    return (
      Object.keys(keys)
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        .filter((k) => keys[k])
        .sort()
    );
  }
}

export default class CompareRunUtil {
  /**
   * Find in a list of metrics/params a certain key
   */
  static findInList(data, key) {
    let found = undefined;
    data.forEach((value) => {
      if (value.key === key) {
        found = value;
      }
    });
    return found;
  }

  /**
   * Get all keys present in the data in ParamLists or MetricLists or Schema input and outputs lists
   */
  static getKeys(lists, numeric) {
    const keys = {};
    lists.forEach((list) =>
      list.forEach((item) => {
        if (!(item.key in keys)) {
          keys[item.key] = true;
        }
        if (numeric && isNaN(parseFloat(item.value))) {
          keys[item.key] = false;
        }
      }),
    );
    return Object.keys(keys)
      .filter((k) => keys[k])
      .sort();
  }
}

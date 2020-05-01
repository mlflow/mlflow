/**
 * A parsed filter used to represent both the parameter and metric key filters in the UI.
 * We currently represent each filter as a list of keys to include, in the order the user wants to
 * display them, and with the empty list meaning "include all keys in sorted order". However, we
 * might want to switch to a more sophisticated filtering language in the future, such as allowing
 * a list of wildcard expressions. In any case, the apply method applies the filter to a list of
 * keys and returns the passing ones in the order we want to display them.
 *
 * NOTE: This class should stay immutable because it's part of some React components' state.
 */
export default class KeyFilter {
  constructor(filterString) {
    this.filterString = filterString || '';
    this.keyList = [];
    this.filterString.split(',').forEach((key) => {
      if (key.trim() !== '') {
        this.keyList.push(key.trim());
      }
    });
  }

  /** Return the filter string originally provided by the user. */
  getFilterString() {
    return this.filterString;
  }

  /**
   * Apply the filter to an array of keys, returning a new array with the keys that pass the filter
   * in the order the user wants them displayed.
   */
  apply(inputKeys) {
    if (this.keyList.length === 0) {
      return inputKeys.slice().sort(); // Just return them in sorted order
    } else {
      const inputKeysAsSet = new Set(inputKeys);
      const outputKeys = [];
      this.keyList.forEach((key) => {
        if (inputKeysAsSet.has(key)) {
          outputKeys.push(key);
        }
      });
      return outputKeys;
    }
  }
}

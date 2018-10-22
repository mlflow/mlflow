/**
 * Class used to represent bagged/non-bagged metrics and params. Unbagged metrics & params are
 * stored in an array to preserve ordering, and other metrics are assumed to be bagged.
 */
class BaggedArray {

  constructor(array) {
    this.array = array;
    this.splitOut = [];
  }

  // Mark a column as "bagged"
  onAddBagged(colName) {
    const idx = this.splitOut.indexOf(colName);
    this.splitOut = idx >= 0 ? this.splitOut.splice(idx, 1) : this.splitOut;
  }

  // Split out a column (remove it from bagged cols)
  onRemoveBagged(colName) {
    this.splitOut.push(colName);

  }

  onSetAllBagged() {
    console.log("Set all bagged!");
    this.splitOut = [];
  }

  onRemoveAllBagged() {
    this.splitOut = this.array.slice(0, this.array.length);
    console.log("Remove all bagged! New array length " + this.splitOut.length + ", full array length: " + this.array.length);
  }

  getBagged() {
    const splitOutSet = new Set(this.splitOut);
    return this.array.filter((elem) => !splitOutSet.has(elem))
  }

  getUnbagged() {
    return this.splitOut.slice(0, this.splitOut.length - 1);
  }

  size() {
    return this.array.length;
  }
}

export default class BaggedArrayUtils {

  // Mark a column as "bagged"
  static withAddBagged(unbagged, colName) {
    const idx = unbagged.indexOf(colName);
    return idx >= 0 ? unbagged.splice(idx, 1) : unbagged;
  }

  // Split out a column (remove it from bagged cols)
  static withRemoveBagged(unbagged, colName) {
    return unbagged.concat([colName]);
  }

  static getUnbagged(bagged, allElems) {
    const baggedSet = new Set(bagged);
    return allElems.filter((elem) => !baggedSet.has(elem))
  }
};
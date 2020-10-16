/* eslint-disable */

/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 12-21-2016
 *   This file is generated. For now, it is a snapshot of the proto messages as of
 *   Jul 23, 2020 4:21:44 PM. We will update the generation pipeline to actually
 *   place these generated records in the correct location shortly.
 */

import Immutable from 'immutable';
import { RecordUtils } from '../../common/sdk/RecordUtils';
import { ModelBuilder } from '../../common/sdk/ModelBuilder';


export const RegisteredModelTag = Immutable.Record({
  // optional STRING
  key: undefined,

  // optional STRING
  value: undefined,
}, 'RegisteredModelTag');

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
RegisteredModelTag.fromJsReviver = function fromJsReviver(key, value) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_RegisteredModelTag = ModelBuilder.extend(RegisteredModelTag, {

  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = RegisteredModelTag.fromJs(pojo);
 */
RegisteredModelTag.fromJs = function fromJs(pojo) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, RegisteredModelTag.fromJsReviver);
  return new extended_RegisteredModelTag(pojoWithNestedImmutables);
};

export const ModelVersionTag = Immutable.Record({
  // optional STRING
  key: undefined,

  // optional STRING
  value: undefined,
}, 'ModelVersionTag');

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
ModelVersionTag.fromJsReviver = function fromJsReviver(key, value) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_ModelVersionTag = ModelBuilder.extend(ModelVersionTag, {

  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = ModelVersionTag.fromJs(pojo);
 */
ModelVersionTag.fromJs = function fromJs(pojo) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, ModelVersionTag.fromJsReviver);
  return new extended_ModelVersionTag(pojoWithNestedImmutables);
};

/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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

export const RegisteredModelTag = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional STRING
    value: undefined,
  },
  'RegisteredModelTag',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(RegisteredModelTag as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
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
(RegisteredModelTag as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (RegisteredModelTag as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_RegisteredModelTag(pojoWithNestedImmutables);
};

export const ModelVersionTag = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional STRING
    value: undefined,
  },
  'ModelVersionTag',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(ModelVersionTag as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
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
(ModelVersionTag as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (ModelVersionTag as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_ModelVersionTag(pojoWithNestedImmutables);
};

/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import type { Fulfilled, Pending, Rejected } from '../../redux-types';

export const isPendingApi = (action: any) => {
  return action.type.endsWith('_PENDING');
};

export const pending = <T extends string>(apiActionType: T): Pending<T> => {
  return `${apiActionType}_PENDING`;
};

export const isFulfilledApi = (action: any) => {
  return action.type.endsWith('_FULFILLED');
};

export const fulfilled = <T extends string>(apiActionType: T): Fulfilled<T> => {
  return `${apiActionType}_FULFILLED`;
};

export const isRejectedApi = (action: any) => {
  return action.type.endsWith('_REJECTED');
};

export const rejected = <T extends string>(apiActionType: any): Rejected<T> => {
  return `${apiActionType}_REJECTED`;
};

export const getUUID = () => {
  const randomPart = Math.random().toString(36).substring(2, 10);
  return new Date().getTime() + randomPart;
};

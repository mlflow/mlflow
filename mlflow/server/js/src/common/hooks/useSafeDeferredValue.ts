import { identity, isFunction } from 'lodash';
import React from 'react';

/**
 * A safe version of `useDeferredValue` that falls back to identity (A->A) if `useDeferredValue` is not supported
 * by current React version.
 */
export const useSafeDeferredValue: <T>(value: T) => T =
  'useDeferredValue' in React && isFunction(React.useDeferredValue) ? React.useDeferredValue : identity;

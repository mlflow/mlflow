import $ from 'jquery';

export const NOOP = () => {};

export function deepFreeze(o) {
  Object.freeze(o);
  Object.getOwnPropertyNames(o).forEach((prop) => {
    if (
      o.hasOwnProperty(prop) &&
      o[prop] !== null &&
      (typeof o[prop] === 'object' || typeof o[prop] === 'function') &&
      !Object.isFrozen(o[prop])
    ) {
      deepFreeze(o[prop]);
    }
  });
  return o;
}

// Replaces AJAX calls with a mock function that returns a nonsense result. This is sufficient
// for testing connected components that would otherwise attempt to make real REST calls, but
// which have no reducers associated -- so the result of the call is thrown out. We mostly
// just want to prevent actual connection attempts from being made and throwing spurious errors.
export function mockAjax() {
  $.ajax = jest.fn().mockImplementation(() => {
    return Promise.resolve({ value: '' });
  });
}

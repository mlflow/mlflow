import type { Expectation } from './ModelTrace.types';

export const getParsedExpectationValue = (expectation: Expectation) => {
  if ('value' in expectation) {
    return expectation.value;
  }

  if (!expectation.serialized_value) {
    return null;
  }

  try {
    // JSON_FORMAT is currently the only supported serialization format.
    return JSON.parse(expectation.serialized_value.value);
  } catch {
    return expectation.serialized_value.value;
  }
};

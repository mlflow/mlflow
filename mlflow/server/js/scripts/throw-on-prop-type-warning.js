// Based on: https://medium.com/shark-bytes/type-checking-with-prop-types-in-jest-e0cd0dc92d5
const originalConsoleError = console.error;

console.error = (message, ...optionalParams) => {
  if (/^Warning: Failed prop type: Invalid prop/.test(message)) {
    throw new Error(message);
  }

  originalConsoleError(message, ...optionalParams);
};

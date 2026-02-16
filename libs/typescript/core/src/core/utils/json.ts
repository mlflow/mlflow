import fastStringify from 'fast-safe-stringify';

// Configure json-bigint to handle large integers

type JSON = {
  parse: (text: string) => any;
  stringify: (value: any) => string;
};

// eslint-disable-next-line @typescript-eslint/no-var-requires
const JSONBigInt = require('./json-bigint/index.js');
// eslint-disable-next-line @typescript-eslint/no-unsafe-call
const JSONBig: JSON = JSONBigInt({
  useNativeBigInt: true,
  alwaysParseAsBig: false,
  storeAsString: false,
});

/**
 * Safely stringify a value to JSON, handling circular references and non-serializable objects.
 * Uses fast-safe-stringify for optimal performance and reliability.
 *
 * @param value The value to stringify
 * @returns JSON string representation of the value
 */
export function safeJsonStringify(value: any): string {
  // MLflow-specific replacer that handles functions, undefined, and errors
  const mlflowReplacer = (_key: string, val: any): any => {
    if (typeof val === 'function') {
      return '[Function]';
    }
    if (val === undefined) {
      return '[Undefined]';
    }

    if (val instanceof Error) {
      return {
        name: val.name,
        message: val.message,
        stack: val.stack,
      };
    }

    return val;
  };

  // Use fast-safe-stringify with our MLflow replacer
  // The circular reference detection is handled by fast-safe-stringify itself
  return fastStringify(value, mlflowReplacer);
}

export { JSONBig };

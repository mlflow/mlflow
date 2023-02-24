import { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';

type PersistSearchSerializeFunctions<Serialized = any, Unserialized = any> = {
  serializeLocalStorage?(input: Unserialized): Serialized;
  serializeQueryString?(input: Unserialized): Serialized;
  deserializeLocalStorage?(input: Serialized): Unserialized;
  deserializeQueryString?(input: Serialized): Unserialized;
};

/**
 * All known field serialization and deserialization mechanisms used in search facets state persisting mechanism.
 */
const persistSearchStateFieldSerializers: Record<string, PersistSearchSerializeFunctions> = {
  /**
   * Array of visible configured charts are serialized into base64-encoded JSON when put into query string
   */
  compareRunCharts: {
    serializeQueryString(input: SearchExperimentRunsFacetsState['compareRunCharts']) {
      return btoa(JSON.stringify(input));
    },
    deserializeQueryString(input: string): SearchExperimentRunsFacetsState['compareRunCharts'] {
      try {
        // Process the URL defensively against intended and unintended malformation
        const parsedResult = JSON.parse(atob(input));
        if (!Array.isArray(parsedResult)) {
          return undefined;
        }
        return parsedResult;
      } catch {
        return undefined;
      }
    },
  },
  /**
   * For "isComparingRuns", we will always save "false" value to local storage so users will
   * get back to default view after visiting the view once more
   */
  isComparingRuns: {
    serializeLocalStorage() {
      return false;
    },
  },
};

type StateKey = keyof Partial<SearchExperimentRunsFacetsState>;

/**
 * Consumes an object with persistable search facets and transforms relevant fields
 * with the registered serialization functions specific to query string.
 * Example scenario: serializing an array of visible configured charts into base64-encoded JSON.
 */
export const serializeFieldsToQueryString = (input: Partial<SearchExperimentRunsFacetsState>) => {
  const resultObject: Partial<Record<StateKey, any>> = { ...input };
  for (const field of Object.keys(resultObject) as StateKey[]) {
    const serializeFn = persistSearchStateFieldSerializers[field]?.serializeQueryString;
    if (serializeFn) {
      resultObject[field] = serializeFn(resultObject[field]);
    }
  }
  return resultObject;
};

/**
 * Consumes an object with search facets extracted from query string and transforms relevant fields
 * with the registered deserialization functions. Example scenario: deserializing an array of
 * visible configured charts from base64-encoded JSON.
 */
export const deserializeFieldsFromQueryString = (
  input: Partial<SearchExperimentRunsFacetsState> | Record<string, any>,
) => {
  const resultObject: Partial<Record<StateKey, any>> = { ...input };
  for (const field of Object.keys(resultObject) as StateKey[]) {
    const deserializeFn = persistSearchStateFieldSerializers[field]?.deserializeQueryString;
    if (deserializeFn) {
      resultObject[field] = deserializeFn(resultObject[field]);
    }
  }
  return resultObject;
};

/**
 * Consumes an object with persistable search facets and transforms relevant fields
 * with the registered serialization functions specific to local storage.
 * Example scenario: serializing an array of visible configured charts into base64-encoded JSON.
 */
export const serializeFieldsToLocalStorage = (input: Partial<SearchExperimentRunsFacetsState>) => {
  const resultObject: Partial<Record<StateKey, any>> = { ...input };
  for (const field of Object.keys(resultObject) as StateKey[]) {
    const serializeFn = persistSearchStateFieldSerializers[field]?.serializeLocalStorage;
    if (serializeFn) {
      resultObject[field] = serializeFn(resultObject[field]);
    }
  }
  return resultObject;
};

/**
 * Consumes an object with search facets extracted from local storage and transforms relevant fields
 * with the registered deserialization functions. Example scenario: deserializing an array of
 * visible configured charts from base64-encoded JSON.
 */
export const deserializeFieldsFromLocalStorage = (
  input: Partial<SearchExperimentRunsFacetsState> | Record<string, any>,
) => {
  const resultObject: Partial<Record<StateKey, any>> = { ...input };
  for (const field of Object.keys(resultObject) as StateKey[]) {
    const deserializeFn = persistSearchStateFieldSerializers[field]?.deserializeLocalStorage;
    if (deserializeFn) {
      resultObject[field] = deserializeFn(resultObject[field]);
    }
  }
  return resultObject;
};

import { isArray } from 'lodash';
import { atobUtf8, btoaUtf8 } from '../../../../common/utils/StringUtils';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';

type PersistSearchSerializeFunctions<Serialized = any, Unserialized = any> = {
  serializeLocalStorage?(input: Unserialized): Serialized;
  serializeQueryString?(input: Unserialized): Serialized;
  deserializeLocalStorage?(input: Serialized): Unserialized;
  deserializeQueryString?(input: Serialized): Unserialized;
};

/**
 * "Flattens" the strings array, i.e. merges it into a single value
 */
const flattenString = (input: string | string[]) => (isArray(input) ? input.join() : input);

/**
 * All known field serialization and deserialization mechanisms used in search facets state persisting mechanism.
 */
const persistSearchStateFieldSerializers: Record<string, PersistSearchSerializeFunctions> = {
  /**
   * In rare cases, search filter might contain commas that interfere with `querystring` library
   * parsing causing it to return array instead of string. Since it's difficult to selectively
   * change `querystring`'s parsing action, we are making sure that the parsed values are always strings.
   */
  searchFilter: {
    deserializeLocalStorage: flattenString,
    deserializeQueryString: flattenString,
  },
  orderByAsc: {
    serializeQueryString(input: boolean) {
      return input.toString();
    },
    deserializeQueryString(input: string) {
      return input === 'true';
    },
  },
  datasetsFilter: {
    serializeQueryString(inputs: ExperimentPageSearchFacetsState['datasetsFilter']) {
      const inputsWithoutExperimentId = inputs.map(({ name, digest, context }) => ({
        name,
        digest,
        context,
      }));
      return btoaUtf8(JSON.stringify(inputsWithoutExperimentId));
    },
    deserializeQueryString(input: string): ExperimentPageSearchFacetsState['datasetsFilter'] {
      try {
        // Process the URL defensively against intended and unintended malformation
        const parsedResult = JSON.parse(atobUtf8(input));
        if (!Array.isArray(parsedResult)) {
          return [];
        }
        return parsedResult;
      } catch {
        return [];
      }
    },
  },
  /**
   * Array of visible configured charts are serialized into base64-encoded JSON when put into query string
   */
  compareRunCharts: {
    serializeQueryString(input: ExperimentPageUIState['compareRunCharts']) {
      return btoaUtf8(JSON.stringify(input));
    },
    deserializeQueryString(input: string): ExperimentPageUIState['compareRunCharts'] {
      try {
        // Process the URL defensively against intended and unintended malformation
        const parsedResult = JSON.parse(atobUtf8(input));
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
   * For "compareRunsMode", we will always save "undefined" value to local storage so users will
   * get back to default view after visiting the view once more.
   */
  compareRunsMode: {
    serializeLocalStorage() {
      return undefined;
    },
  },
};

type StateKey = keyof Partial<ExperimentPageSearchFacetsState>;

/**
 * Consumes an object with persistable search facets and transforms relevant fields
 * with the registered serialization functions specific to query string.
 * Example scenario: serializing an array of visible configured charts into base64-encoded JSON.
 */
export const serializeFieldsToQueryString = (input: Partial<ExperimentPageSearchFacetsState>) => {
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
  input: Partial<ExperimentPageSearchFacetsState> | Record<string, any>,
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
export const serializeFieldsToLocalStorage = (input: Partial<ExperimentPageSearchFacetsState>) => {
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
  input: Partial<ExperimentPageSearchFacetsState> | Record<string, any>,
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

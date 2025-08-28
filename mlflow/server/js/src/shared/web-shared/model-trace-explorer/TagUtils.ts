import _ from 'lodash';

export const MLFLOW_INTERNAL_PREFIX = 'mlflow.';
const MLFLOW_INTERNAL_PREFIX_UC = '_mlflow_';

export const isUserFacingTag = (tagKey: string) =>
  !tagKey.startsWith(MLFLOW_INTERNAL_PREFIX) && !tagKey.startsWith(MLFLOW_INTERNAL_PREFIX_UC);

// Safe JSON.parse that returns undefined instead of throwing an error
export const parseJSONSafe = (json: string) => {
  try {
    return JSON.parse(json);
  } catch (e) {
    return undefined;
  }
};

export const truncateToFirstLineWithMaxLength = (str: string, maxLength: number): string => {
  const truncated = _.truncate(str, {
    length: maxLength,
  });
  return _.takeWhile(truncated, (char) => char !== '\n').join('');
};

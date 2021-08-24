import _ from 'lodash';

export const truncateToFirstLineWithMaxLength = (str, maxLength) => {
  const truncated = _.truncate(str, {
    length: maxLength,
  });
  return _.takeWhile(truncated, (char) => char !== '\n').join('');
};

export const capitalizeFirstChar = (str) => {
  if (!str || typeof str !== 'string' || str.length < 1) {
    return str;
  }
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

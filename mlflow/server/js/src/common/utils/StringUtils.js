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

export const middleTruncateStr = (str, maxLen) => {
  if (str.length > maxLen) {
    const firstPartLen = Math.floor((maxLen - 3) / 2);
    const lastPartLen = maxLen - 3 - firstPartLen;
    return (
      str.substring(0, firstPartLen) + '...' + str.substring(str.length - lastPartLen, str.length)
    );
  } else {
    return str;
  }
};

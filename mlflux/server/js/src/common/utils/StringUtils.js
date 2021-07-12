import _ from 'lodash';

export const truncateToFirstLineWithMaxLength = (str, maxLength) => {
  const truncated = _.truncate(str, {
    length: maxLength,
  });
  return _.takeWhile(truncated, (char) => char !== '\n').join('');
};

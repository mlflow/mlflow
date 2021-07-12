import { truncateToFirstLineWithMaxLength } from './StringUtils';

describe('truncateToFirstLineWithMaxLength', () => {
  test('should truncate to first line if it exists', () => {
    const str = 'Test string\nwith a new line';
    expect(truncateToFirstLineWithMaxLength(str, 32)).toEqual('Test string');
  });

  test('if first line longer than maxLength, should truncate and add ellipses', () => {
    const str = 'This is 24 characters, so this part should be truncated';
    expect(truncateToFirstLineWithMaxLength(str, 24)).toEqual('This is 24 characters...');
  });

  test('should not add ellipses if length is equal to maxLength', () => {
    const str = 'This is 21 characters';
    expect(truncateToFirstLineWithMaxLength(str, 21)).toEqual(str);
  });

  test('should not truncate if only 1 line that is shorter than maxLength', () => {
    const str = 'A short line';
    expect(truncateToFirstLineWithMaxLength(str, 32)).toEqual(str);
  });
});

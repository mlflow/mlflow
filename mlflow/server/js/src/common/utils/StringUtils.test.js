import { truncateToFirstLineWithMaxLength, capitalizeFirstChar } from './StringUtils';

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

describe('capitalizeFirstChar', () => {
  test('should capitalize first char and lower case all other chars', () => {
    const str = 'i WaNt THis tO oNlY cAPItaLize FirSt ChaR.';
    expect(capitalizeFirstChar(str)).toEqual('I want this to only capitalize first char.');
  });

  test('should not work for str with length less than 1', () => {
    const str = '';
    expect(capitalizeFirstChar(str)).toEqual(str);
  });

  test('should not work for objects that are not string', () => {
    const number = 2;
    const array = ['not', 'work'];
    const object = { key: 'value' };
    expect(capitalizeFirstChar(null)).toEqual(null);
    expect(capitalizeFirstChar(number)).toEqual(number);
    expect(capitalizeFirstChar(array)).toEqual(array);
    expect(capitalizeFirstChar(object)).toEqual(object);
  });
});

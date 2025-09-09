import {
  truncateToFirstLineWithMaxLength,
  capitalizeFirstChar,
  middleTruncateStr,
  btoaUtf8,
  atobUtf8,
  isTextCompressedDeflate,
  textCompressDeflate,
  textDecompressDeflate,
} from './StringUtils';

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

describe('middleTruncateStr', () => {
  test('middleTruncateStr', () => {
    expect(middleTruncateStr('abc', 10)).toEqual('abc');
    expect(middleTruncateStr('abcdefghij', 10)).toEqual('abcdefghij');
    expect(middleTruncateStr('abcdefghijk', 10)).toEqual('abc...hijk');
    expect(middleTruncateStr('abcdefghijkl', 10)).toEqual('abc...ijkl');
  });
});

describe('btoaUtf8 and atobUtf8', () => {
  test('decodes and encodes a simple ascii string', () => {
    const testString = 'abcdefghi[]=__---11123';
    expect(btoaUtf8(testString)).toEqual(btoa(testString));
    expect(atob(btoaUtf8(testString))).toEqual(testString);
    expect(atobUtf8(btoaUtf8(testString))).toEqual(testString);
  });

  test('decodes and encodes a serialized JSON object in a way compatible to stock btoa()', () => {
    const testObject = {
      number: 123,
      text: '123',
      array: [1, 3, { some: 'object', nested: ['array'] }],
    };
    const stringifiedJson = JSON.stringify(testObject);
    expect(btoaUtf8(stringifiedJson)).toEqual(btoa(stringifiedJson));
    expect(atob(btoaUtf8(stringifiedJson))).toEqual(stringifiedJson);
    expect(atobUtf8(btoaUtf8(stringifiedJson))).toEqual(stringifiedJson);
    expect(JSON.parse(atobUtf8(btoaUtf8(stringifiedJson)))).toEqual(testObject);
  });

  test('decodes and encodes an utf-8 ascii string', () => {
    const testString = 'abcdef#ąółźļż👀中文';
    expect(() => btoa(testString)).toThrow();
    expect(() => btoaUtf8(testString)).not.toThrow();
    expect(atobUtf8(btoaUtf8(testString))).toEqual(testString);
  });

  test('decodes and encodes a serialized JSON object with utf-8 characters', () => {
    const testObject = {
      number: 123,
      中文: '123',
      ʎɐɹɹɐ: [1, 3, { some: '👀bject', nested: ['ᴀʀʀᴀʏ'] }],
    };
    const stringifiedJson = JSON.stringify(testObject);
    expect(atobUtf8(btoaUtf8(stringifiedJson))).toEqual(stringifiedJson);
    expect(JSON.parse(atobUtf8(btoaUtf8(stringifiedJson)))).toEqual(testObject);
  });

  test('handles empty values', () => {
    expect(btoaUtf8('')).toEqual('');
    expect(atobUtf8('')).toEqual('');
  });
});

const testCompressedObject = {
  viewable_object_ids: ['7b3e00a6-6459-4ea6-97b9-9fb58f0265bc'],
  viewable_objects: {
    '7b3e00a6-6459-4ea6-97b9-9fb58f0265bc': { id: '7b3e00a6-6459-4ea6-97b9-9fb58f0265bc', name: 'test' },
  },
};

describe('text compression utils', () => {
  test.each([
    { text: 'hello world', name: 'simple' },
    { text: 'ąółźļż👀中文中文👀中文', name: 'some unicode' },
    { text: JSON.stringify(testCompressedObject), name: 'complex' },
    {
      text: '\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff',
      name: 'raw unicode',
    },
  ])('deflate and inflate text ($name)', async ({ text }) => {
    const compressed = await textCompressDeflate(text);
    expect(isTextCompressedDeflate(compressed)).toEqual(true);

    const decompressed = await textDecompressDeflate(compressed);
    expect(decompressed).toEqual(text);
  });

  test('should throw error when decompressing invalid compressed text', async () => {
    const compressed = 'xyz;invalid';
    await expect(textDecompressDeflate(compressed)).rejects.toThrow('Invalid compressed text, payload header invalid');
  });
});

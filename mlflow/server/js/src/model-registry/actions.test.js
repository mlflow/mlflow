import { resolveFilterValue } from './actions';

describe('action tests', () => {
  test('simple string', () => {
    expect(resolveFilterValue('hello')).toEqual("'hello'");
  });

  test('simple string with wildcard', () => {
    expect(resolveFilterValue('hello', true)).toEqual("'%hello%'");
  });

  test('simple string spaces', () => {
    expect(resolveFilterValue(' he llo  ')).toEqual("' he llo  '");
  });

  test('simple string spaces with wildcard', () => {
    expect(resolveFilterValue(' he llo  ', true)).toEqual("'% he llo  %'");
  });

  test('single quotes', () => {
    expect(resolveFilterValue("A's model")).toEqual('"A\'s model"');
  });

  test('single quotes with wildcard', () => {
    expect(resolveFilterValue("A's model", true)).toEqual('"%A\'s model%"');
  });

  test('double quotes', () => {
    expect(resolveFilterValue('the "best" model')).toEqual('\'the "best" model\'');
  });

  test('double quotes with wildcard', () => {
    expect(resolveFilterValue('the "best" model', true)).toEqual('\'%the "best" model%\'');
  });

  test('percent character', () => {
    expect(resolveFilterValue('%')).toEqual("'%'");
  });

  test('percent character with wildcard', () => {
    expect(resolveFilterValue('%', true)).toEqual("'%%%'");
  });
});

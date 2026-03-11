import { isValidTagKey } from './tagKeyValidation';

describe('isValidTagKey', () => {
  it('allows empty string', () => {
    expect(isValidTagKey('')).toBe(true);
  });

  it('allows alphanumeric and underscore', () => {
    expect(isValidTagKey('key')).toBe(true);
    expect(isValidTagKey('key_1')).toBe(true);
    expect(isValidTagKey('env2')).toBe(true);
  });

  it('allows slash in the middle (aligned with backend)', () => {
    expect(isValidTagKey('env/prod')).toBe(true);
    expect(isValidTagKey('team/name')).toBe(true);
  });

  it('allows period, dash, space, colon (aligned with backend)', () => {
    expect(isValidTagKey('my.key')).toBe(true);
    expect(isValidTagKey('my-key')).toBe(true);
    expect(isValidTagKey('my key')).toBe(true);
    expect(isValidTagKey('key:value')).toBe(true);
  });

  it('rejects leading slash (path_not_unique)', () => {
    expect(isValidTagKey('/key')).toBe(false);
    expect(isValidTagKey('/')).toBe(false);
  });

  it('rejects single dot', () => {
    expect(isValidTagKey('.')).toBe(false);
  });

  it('rejects starting with ..', () => {
    expect(isValidTagKey('..')).toBe(false);
    expect(isValidTagKey('../key')).toBe(false);
  });

  it('allows path-like keys (e.g. test/.test); backend validates path rules and returns error if invalid', () => {
    expect(isValidTagKey('test/.test')).toBe(true);
    expect(isValidTagKey('a/./b')).toBe(true);
  });

  it('rejects characters not allowed by backend', () => {
    expect(isValidTagKey('key,value')).toBe(false);
    expect(isValidTagKey('key=value')).toBe(false);
    expect(isValidTagKey('key@domain')).toBe(false);
  });
});

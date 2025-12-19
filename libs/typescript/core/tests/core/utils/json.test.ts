import { safeJsonStringify } from '../../../src/core/utils/json';

describe('safeJsonStringify', () => {
  it('should stringify simple values correctly', () => {
    expect(safeJsonStringify('hello')).toBe('"hello"');
    expect(safeJsonStringify(123)).toBe('123');
    expect(safeJsonStringify(true)).toBe('true');
    expect(safeJsonStringify(null)).toBe('null');
    expect(safeJsonStringify({ a: 1, b: 'test' })).toBe('{"a":1,"b":"test"}');
    expect(safeJsonStringify([1, 2, 3])).toBe('[1,2,3]');
  });

  it('should handle circular references', () => {
    const obj: any = { name: 'test' };
    obj.self = obj;

    const result = safeJsonStringify(obj);
    const parsed = JSON.parse(result);

    expect(parsed.name).toBe('test');
    expect(parsed.self).toBe('[Circular]');
  });

  it('should handle functions', () => {
    const obj = {
      name: 'test',
      fn: () => console.log('hello'),
      method: function () {
        return 42;
      }
    };

    const result = safeJsonStringify(obj);
    const parsed = JSON.parse(result);

    expect(parsed.name).toBe('test');
    expect(parsed.fn).toBe('[Function]');
    expect(parsed.method).toBe('[Function]');
  });

  it('should handle undefined values', () => {
    const obj = {
      name: 'test',
      value: undefined,
      nested: { prop: undefined }
    };

    const result = safeJsonStringify(obj);
    const parsed = JSON.parse(result);

    expect(parsed.name).toBe('test');
    expect(parsed.value).toBe('[Undefined]');
    expect(parsed.nested.prop).toBe('[Undefined]');
  });

  it('should handle Error objects', () => {
    const error = new Error('Test error');
    const obj = {
      status: 'failed',
      error
    };

    const result = safeJsonStringify(obj);
    const parsed = JSON.parse(result);

    expect(parsed.status).toBe('failed');
    expect(parsed.error.name).toBe('Error');
    expect(parsed.error.message).toBe('Test error');
    expect(parsed.error.stack).toBeDefined();
  });
});

import {
  deserializeFieldsFromLocalStorage,
  deserializeFieldsFromQueryString,
  serializeFieldsToLocalStorage,
  serializeFieldsToQueryString,
} from './persistSearchFacets.serializers';

describe('persistSearchFacets serializers and deserializers', () => {
  it('tests serializeToQueryString', () => {
    const serializedObject = serializeFieldsToQueryString({
      orderByKey: 'column_name',
    });

    expect(serializedObject).toEqual(
      expect.objectContaining({
        orderByKey: 'column_name',
      }),
    );
  });

  it('tests deserializeToQueryString', () => {
    const deserializedObject = deserializeFieldsFromQueryString({
      orderByKey: 'column_name',
    });

    expect(deserializedObject).toEqual(
      expect.objectContaining({
        orderByKey: 'column_name',
      }),
    );
  });

  it('tests serializeLocalStorage', () => {
    const serializedObject = serializeFieldsToLocalStorage({
      orderByKey: 'column_name',
    });

    expect(serializedObject).toEqual(
      expect.objectContaining({
        orderByKey: 'column_name',
      }),
    );
  });

  it('tests deserializing search filter without extra characters', () => {
    const serializedObjectQs = deserializeFieldsFromQueryString({
      searchFilter: ['param.p1 = "something', 'separated', 'by comma"'],
    });
    const serializedObjectLs = deserializeFieldsFromLocalStorage({
      searchFilter: ['param.p1 = "something', 'separated', 'by comma"'],
    });

    expect(serializedObjectQs).toEqual(serializedObjectLs);
    expect(serializedObjectQs).toEqual(
      expect.objectContaining({
        searchFilter: 'param.p1 = "something,separated,by comma"',
      }),
    );
  });

  describe('runLimit and hideFinishedRuns serialization', () => {
    it('serializes runLimit to query string', () => {
      const serialized = serializeFieldsToQueryString({
        runLimit: 10,
      });

      expect(serialized).toEqual(
        expect.objectContaining({
          runLimit: '10',
        }),
      );
    });

    it('omits runLimit when null in query string', () => {
      const serialized = serializeFieldsToQueryString({
        runLimit: null,
      });

      // Should not include runLimit property when it's null
      expect(serialized.runLimit).toBeUndefined();
    });

    it('deserializes runLimit from query string', () => {
      const deserialized = deserializeFieldsFromQueryString({
        runLimit: '20',
      });

      expect(deserialized).toEqual(
        expect.objectContaining({
          runLimit: 20,
        }),
      );
    });

    it('serializes hideFinishedRuns to query string', () => {
      const serialized = serializeFieldsToQueryString({
        hideFinishedRuns: true,
      });

      expect(serialized).toEqual(
        expect.objectContaining({
          hideFinishedRuns: 'true',
        }),
      );
    });

    it('deserializes hideFinishedRuns from query string', () => {
      const deserialized = deserializeFieldsFromQueryString({
        hideFinishedRuns: 'true',
      });

      expect(deserialized).toEqual(
        expect.objectContaining({
          hideFinishedRuns: true,
        }),
      );
    });

    it('serializes both runLimit and hideFinishedRuns to local storage', () => {
      const serialized = serializeFieldsToLocalStorage({
        runLimit: 15,
        hideFinishedRuns: false,
      });

      expect(serialized).toEqual(
        expect.objectContaining({
          runLimit: 15,
          hideFinishedRuns: false,
        }),
      );
    });

    it('deserializes both runLimit and hideFinishedRuns from local storage', () => {
      const deserialized = deserializeFieldsFromLocalStorage({
        runLimit: 25,
        hideFinishedRuns: true,
      });

      expect(deserialized).toEqual(
        expect.objectContaining({
          runLimit: 25,
          hideFinishedRuns: true,
        }),
      );
    });

    it('handles malformed runLimit values in query string', () => {
      // Test various malformed runLimit values
      expect(deserializeFieldsFromQueryString({ runLimit: '' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: 'invalid' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: 'foo' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: 'NaN' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: '-5' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: '0' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: '1.5' }).runLimit).toBe(null);
      expect(deserializeFieldsFromQueryString({ runLimit: 'Infinity' }).runLimit).toBe(null);
    });
  });
});

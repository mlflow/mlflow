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
});

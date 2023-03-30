import {
  deserializeFieldsFromQueryString,
  serializeFieldsToLocalStorage,
  serializeFieldsToQueryString,
} from './persistSearchFacets.serializers';

describe('persistSearchFacets serializers and deserializers', () => {
  it('tests serializeToQueryString', () => {
    const serializedObject = serializeFieldsToQueryString({
      orderByKey: 'column_name',
      compareRunCharts: [{ metricKey: 'metric', uuid: 'abc-123' } as any],
    });

    expect(serializedObject).toEqual(
      expect.objectContaining({
        orderByKey: 'column_name',
        compareRunCharts: 'W3sibWV0cmljS2V5IjoibWV0cmljIiwidXVpZCI6ImFiYy0xMjMifV0=',
      }),
    );
  });

  it('tests deserializeToQueryString', () => {
    const deserializedObject = deserializeFieldsFromQueryString({
      orderByKey: 'column_name',
      compareRunCharts: 'W3sibWV0cmljS2V5IjoiYW5vdGhlci1tZXRyaWMiLCJ1dWlkIjoiMTIzLWlkIn1d',
    });

    expect(deserializedObject).toEqual(
      expect.objectContaining({
        orderByKey: 'column_name',
        compareRunCharts: [{ metricKey: 'another-metric', uuid: '123-id' }],
      }),
    );
  });

  it('tests deserializeToQueryString with erronoeus data', () => {
    const deserializedObject = deserializeFieldsFromQueryString({
      compareRunCharts: 'something-not-deserializable',
    });

    expect(deserializedObject).toEqual(
      expect.objectContaining({
        compareRunCharts: undefined,
      }),
    );
  });

  it('tests serializeLocalStorage', () => {
    const serializedObject = serializeFieldsToLocalStorage({
      orderByKey: 'column_name',
      isComparingRuns: true,
    });

    expect(serializedObject).toEqual(
      expect.objectContaining({
        orderByKey: 'column_name',
        isComparingRuns: false,
      }),
    );
  });
});

import {
  TraceInfo,
  TraceLocationType,
  TraceState,
  calculateCostByModelAndTokenUsage,
  constructTraceIdV4,
  createTraceLocationFromUcTablePrefix,
  getUcLocationString,
} from '../src';

describe('public UC TraceInfo API', () => {
  it('exports runtime entities and V4 trace helpers', () => {
    const traceLocation = createTraceLocationFromUcTablePrefix('catalog', 'schema', 'prefix');
    const location = getUcLocationString(traceLocation);
    const traceInfo = new TraceInfo({
      traceId: constructTraceIdV4(location!, 'abc123'),
      traceLocation,
      requestTime: 1_700_000_000_000,
      state: TraceState.OK,
    });

    expect(location).toBe('catalog.schema.prefix');
    expect(traceLocation.type).toBe(TraceLocationType.UC_TABLE_PREFIX);
    expect(traceInfo).toBeInstanceOf(TraceInfo);
    expect(traceInfo.traceId).toBe('trace:/catalog.schema.prefix/abc123');
    expect(
      calculateCostByModelAndTokenUsage(
        'gpt-5-mini',
        { input_tokens: 1, output_tokens: 1 },
        'openai',
      )?.total_cost,
    ).toBeGreaterThan(0);
  });
});

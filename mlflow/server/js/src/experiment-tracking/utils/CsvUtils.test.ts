import { chartDataToCsv, chartMetricHistoryToCsv, runInfosToCsv } from './CsvUtils';

const createFakePayload = (n = 3): any => {
  const runNames = new Array(n).fill('').map((_, index) => `run-${index + 1}`);
  return {
    runInfos: runNames.map((name, index) => ({
      runUuid: `uuid-for-${name}`,
      experimentId: '123',
      runName: name,
      status: 'FINISHED',
      startTime: 1669896000000 + index * 60000, // 2022-12-01 12:00:00Z + 1 minute per run index
      endTime: 1669896001000 + index * 60000, // 2022-12-01 12:00:01Z + 1 minute per run index
    })),
    paramKeyList: ['param_1', 'param_2'],
    metricKeyList: ['metric_1', 'metric_2'],
    tagKeyList: ['tag_1'],
    paramsList: runNames.map((name) => [
      { key: 'param_1', value: `param_1_for_${name}` },
      { key: 'param_2', value: `param_2_for_${name}` },
    ]),
    tagsList: runNames.map((name) => ({
      tag_1: { key: 'tag_1', value: `tag_1_for_${name}` },
    })),
    metricsList: runNames.map((_, index) => [
      { key: 'metric_1', value: (index + 1) * 10 + 1 }, // run 1 will get 11, run two will get 21 etc.
      { key: 'metric_2', value: (index + 1) * 10 + 2 }, // run 1 will get 12, run two will get 22 etc.
    ]),
  };
};

describe('CsvUtils', () => {
  it('generates proper number of runs', () => {
    const resultCsv = runInfosToCsv(createFakePayload(10));
    expect(resultCsv.trim().split('\n').length).toEqual(11); // One header line and 10 data lines
  });

  it('generates empty set for empty run list', () => {
    const resultCsv = runInfosToCsv(createFakePayload(0));

    // Assert that it's just the header and nothing else
    expect(resultCsv.trim()).toEqual(
      '"Start Time","Duration","Run ID","Name","Source Type","Source Name","User","Status","param_1","param_2","metric_1","metric_2","tag_1"',
    );
  });

  it('generates valid header and data', () => {
    const resultCsv = runInfosToCsv(createFakePayload(3));
    const csvStrings = resultCsv.trim().split('\n');

    const run1csv = csvStrings[1].split(',');
    const run2csv = csvStrings[2].split(',');
    const run3csv = csvStrings[3].split(',');

    // Assert header contents
    expect(csvStrings[0]).toEqual(
      '"Start Time","Duration","Run ID","Name","Source Type","Source Name","User","Status","param_1","param_2","metric_1","metric_2","tag_1"',
    );

    const PARAM_1_INDEX_POS = 8;
    const PARAM_2_INDEX_POS = 9;
    const METRIC_1_INDEX_POS = 10;
    const METRIC_2_INDEX_POS = 11;
    const TAG_1_INDEX_POS = 12;

    expect(run1csv).toContain('"2022-12-01 12:00:00"');
    expect(run2csv).toContain('"2022-12-01 12:01:00"');
    expect(run3csv).toContain('"2022-12-01 12:02:00"');

    expect(run1csv).toContain('"1.0s"');
    expect(run2csv).toContain('"1.0s"');
    expect(run3csv).toContain('"1.0s"');

    expect(run1csv).toContain('"uuid-for-run-1"');
    expect(run2csv).toContain('"uuid-for-run-2"');
    expect(run3csv).toContain('"uuid-for-run-3"');

    expect(run1csv).toContain('"run-1"');
    expect(run2csv).toContain('"run-2"');
    expect(run3csv).toContain('"run-3"');

    expect(run1csv[PARAM_1_INDEX_POS]).toEqual('"param_1_for_run-1"');
    expect(run2csv[PARAM_1_INDEX_POS]).toEqual('"param_1_for_run-2"');
    expect(run3csv[PARAM_1_INDEX_POS]).toEqual('"param_1_for_run-3"');

    expect(run1csv[PARAM_2_INDEX_POS]).toEqual('"param_2_for_run-1"');
    expect(run2csv[PARAM_2_INDEX_POS]).toEqual('"param_2_for_run-2"');
    expect(run3csv[PARAM_2_INDEX_POS]).toEqual('"param_2_for_run-3"');

    expect(run1csv[METRIC_1_INDEX_POS].toString()).toEqual('"11"');
    expect(run2csv[METRIC_1_INDEX_POS].toString()).toEqual('"21"');
    expect(run3csv[METRIC_1_INDEX_POS].toString()).toEqual('"31"');

    expect(run1csv[METRIC_2_INDEX_POS].toString()).toEqual('"12"');
    expect(run2csv[METRIC_2_INDEX_POS].toString()).toEqual('"22"');
    expect(run3csv[METRIC_2_INDEX_POS].toString()).toEqual('"32"');

    expect(run1csv[TAG_1_INDEX_POS]).toEqual('"tag_1_for_run-1"');
    expect(run2csv[TAG_1_INDEX_POS]).toEqual('"tag_1_for_run-2"');
    expect(run3csv[TAG_1_INDEX_POS]).toEqual('"tag_1_for_run-3"');
  });

  it('generates proper metric history CSV for run traces', () => {
    const traces = [
      {
        displayName: 'Run 1',
        runInfo: { runUuid: 'uuid-1' },
        metricsHistory: {
          metric1: [
            { key: 'metric1', step: 1, timestamp: 1000, value: 10 },
            { key: 'metric1', step: 2, timestamp: 2000, value: 20 },
          ],
          metric2: [
            { key: 'metric2', step: 1, timestamp: 1000, value: 100 },
            { key: 'metric2', step: 2, timestamp: 2000, value: 200 },
          ],
        },
      },
      {
        displayName: 'Run 2',
        runInfo: { runUuid: 'uuid-2' },
        metricsHistory: {
          metric1: [
            { key: 'metric1', step: 1, timestamp: 1000, value: 30 },
            { key: 'metric1', step: 2, timestamp: 2000, value: 40 },
          ],
          metric2: [
            { key: 'metric2', step: 1, timestamp: 1000, value: 300 },
            { key: 'metric2', step: 2, timestamp: 2000, value: 400 },
          ],
        },
      },
    ] as any;

    const metricKeys = ['metric1', 'metric2'];

    const expectedCsv = `"Run","Run ID","metric","step","timestamp","value"
"Run 1","uuid-1","metric1","1","1000","10"
"Run 1","uuid-1","metric1","2","2000","20"
"Run 2","uuid-2","metric1","1","1000","30"
"Run 2","uuid-2","metric1","2","2000","40"
"Run 1","uuid-1","metric2","1","1000","100"
"Run 1","uuid-1","metric2","2","2000","200"
"Run 2","uuid-2","metric2","1","1000","300"
"Run 2","uuid-2","metric2","2","2000","400"`;

    const resultCsv = chartMetricHistoryToCsv(traces, metricKeys);

    expect(resultCsv.trim()).toEqual(expectedCsv);
  });

  it('generates proper CSV for multi-metric and multi-param chart data', () => {
    const traces = [
      {
        displayName: 'Run 1',
        runInfo: { runUuid: 'uuid-1' },
        metrics: {
          metric1: { key: 'metric1', value: 10 },
          metric2: { key: 'metric2', value: 100 },
        },
        params: {
          param1: { key: 'param1', value: 'value1' },
          param2: { key: 'param2', value: 'value2' },
        },
      },
      {
        displayName: 'Run 2',
        runInfo: { runUuid: 'uuid-2' },
        metrics: {
          metric1: { key: 'metric1', value: 20 },
          metric2: { key: 'metric2', value: 200 },
        },
        params: {
          param1: { key: 'param1', value: 'value3' },
          param2: { key: 'param2', value: 'value4' },
        },
      },
    ] as any;

    const metricKeys = ['metric1', 'metric2'];
    const paramKeys = ['param1', 'param2'];

    const expectedCsv = `"Run","Run ID","metric1","metric2","param1","param2"
"Run 1","uuid-1","10","100","value1","value2"
"Run 2","uuid-2","20","200","value3","value4"`;

    const resultCsv = chartDataToCsv(traces, metricKeys, paramKeys);

    expect(resultCsv.trim()).toEqual(expectedCsv);
  });
});

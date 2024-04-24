import { runInfosToCsv } from './CsvUtils';

const createFakePayload = (n = 3): any => {
  const runNames = new Array(n).fill('').map((_, index) => `run-${index + 1}`);
  return {
    runInfos: runNames.map((name, index) => ({
      run_uuid: `uuid-for-${name}`,
      experiment_id: '123',
      run_name: name,
      status: 'FINISHED',
      start_time: 1669896000000 + index * 60000, // 2022-12-01 12:00:00Z + 1 minute per run index
      end_time: 1669896001000 + index * 60000, // 2022-12-01 12:00:01Z + 1 minute per run index
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
      'Start Time,Duration,Run ID,Name,Source Type,Source Name,User,Status,param_1,param_2,metric_1,metric_2,tag_1',
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
      'Start Time,Duration,Run ID,Name,Source Type,Source Name,User,Status,param_1,param_2,metric_1,metric_2,tag_1',
    );

    const PARAM_1_INDEX_POS = 8;
    const PARAM_2_INDEX_POS = 9;
    const METRIC_1_INDEX_POS = 10;
    const METRIC_2_INDEX_POS = 11;
    const TAG_1_INDEX_POS = 12;

    expect(run1csv).toContain('2022-12-01 12:00:00');
    expect(run2csv).toContain('2022-12-01 12:01:00');
    expect(run3csv).toContain('2022-12-01 12:02:00');

    expect(run1csv).toContain('1.0s');
    expect(run2csv).toContain('1.0s');
    expect(run3csv).toContain('1.0s');

    expect(run1csv).toContain('uuid-for-run-1');
    expect(run2csv).toContain('uuid-for-run-2');
    expect(run3csv).toContain('uuid-for-run-3');

    expect(run1csv).toContain('run-1');
    expect(run2csv).toContain('run-2');
    expect(run3csv).toContain('run-3');

    expect(run1csv[PARAM_1_INDEX_POS]).toEqual('param_1_for_run-1');
    expect(run2csv[PARAM_1_INDEX_POS]).toEqual('param_1_for_run-2');
    expect(run3csv[PARAM_1_INDEX_POS]).toEqual('param_1_for_run-3');

    expect(run1csv[PARAM_2_INDEX_POS]).toEqual('param_2_for_run-1');
    expect(run2csv[PARAM_2_INDEX_POS]).toEqual('param_2_for_run-2');
    expect(run3csv[PARAM_2_INDEX_POS]).toEqual('param_2_for_run-3');

    expect(run1csv[METRIC_1_INDEX_POS].toString()).toEqual('11');
    expect(run2csv[METRIC_1_INDEX_POS].toString()).toEqual('21');
    expect(run3csv[METRIC_1_INDEX_POS].toString()).toEqual('31');

    expect(run1csv[METRIC_2_INDEX_POS].toString()).toEqual('12');
    expect(run2csv[METRIC_2_INDEX_POS].toString()).toEqual('22');
    expect(run3csv[METRIC_2_INDEX_POS].toString()).toEqual('32');

    expect(run1csv[TAG_1_INDEX_POS]).toEqual('tag_1_for_run-1');
    expect(run2csv[TAG_1_INDEX_POS]).toEqual('tag_1_for_run-2');
    expect(run3csv[TAG_1_INDEX_POS]).toEqual('tag_1_for_run-3');
  });
});

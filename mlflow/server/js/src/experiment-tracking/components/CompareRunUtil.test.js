import CompareRunUtil from './CompareRunUtil';
import { Metric } from '../sdk/MlflowMessages';

describe('CompareRunUtil', () => {
  const makeSublist = (obj) => {
    const newObj = {};
    Object.entries(obj).forEach(
      ([key, value]) => (newObj[key] = Metric.fromJs({ key: key, value: value })),
    );
    return Object.values(newObj);
  };

  const list = [
    // run 1
    makeSublist({
      rmse: 0.01,
      objective: 37.121,
      timestamp: 1571684492701,
      step: 1,
      strnum: '123',
      comment: 'oh 1',
    }),
    // run 2
    makeSublist({
      rmse: 0.02,
      objective: 37.122,
      timestamp: 1571684492702,
      step: 2,
      comment: 'oh 2',
      strnum: '4.5',
      loss: 1.0783,
    }),
  ];

  test('findInList - properly returns when not in list', () => {
    expect(CompareRunUtil.findInList(list[0], 'loss')).toEqual(undefined);
    expect(CompareRunUtil.findInList(list[0], undefined)).toEqual(undefined);
  });

  test('findInList - finds various value types in list', () => {
    expect(CompareRunUtil.findInList(list[0], 'rmse').value).toEqual(0.01);
    expect(CompareRunUtil.findInList(list[1], 'rmse').value).toEqual(0.02);
    expect(CompareRunUtil.findInList(list[0], 'timestamp').value).toEqual(1571684492701);
    expect(CompareRunUtil.findInList(list[1], 'timestamp').value).toEqual(1571684492702);
    expect(CompareRunUtil.findInList(list[0], 'comment').value).toEqual('oh 1');
    expect(CompareRunUtil.findInList(list[1], 'comment').value).toEqual('oh 2');
    expect(CompareRunUtil.findInList(list[0], 'strnum').value).toEqual('123');
    expect(CompareRunUtil.findInList(list[1], 'strnum').value).toEqual('4.5');
  });

  test('getKeys - no keys', () => {
    const emptyList = [[], []];
    expect(CompareRunUtil.getKeys(emptyList, false)).toEqual([]);
    expect(CompareRunUtil.getKeys(emptyList, true)).toEqual([]);
  });

  test("getKeys - no keys for 'numeric' = true", () => {
    const nonnumericList = [
      // run 1
      makeSublist({
        str1: 'blah',
        str2: 'bleh',
        str3: 'bloh',
      }),
      // run 2
      makeSublist({
        str1: 'ablah',
        str2: 'ableh',
        str4: 'abloh',
      }),
    ];
    expect(CompareRunUtil.getKeys(nonnumericList, true)).toEqual([]);
  });

  test('getKeys - returns keys properly', () => {
    expect(CompareRunUtil.getKeys(list, false)).toEqual([
      'comment',
      'loss',
      'objective',
      'rmse',
      'step',
      'strnum',
      'timestamp',
    ]);
    expect(CompareRunUtil.getKeys(list, true)).toEqual([
      'loss',
      'objective',
      'rmse',
      'step',
      'strnum',
      'timestamp',
    ]);
  });
});

// Note that if you call `jest.useFakeTimers`, it will reset some of these. So if
// you still need performance mocked, add `{ doNotFake: ['performance'] }`!
const fakeEntry: PerformanceMark = {
  duration: 0,
  entryType: 'measure',
  name: 'fake',
  startTime: 0,
  toJSON: () => ({}),
  detail: null,
};

global.performance.clearMarks ??= () => {};
global.performance.clearMeasures ??= () => {};
global.performance.getEntriesByType ??= () => [fakeEntry];
global.performance.getEntriesByName ??= () => [fakeEntry];
global.performance.mark ??= (name) => ({ ...fakeEntry, name });
global.performance.measure ??= (name) => ({ ...fakeEntry, name });
// @ts-expect-error Mocking to avoid failures.
global.performance.timing ??= {};

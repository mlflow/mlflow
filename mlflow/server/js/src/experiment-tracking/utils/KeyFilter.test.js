import KeyFilter from './KeyFilter';

test('basic filter', () => {
  const filter = new KeyFilter('b, a');
  expect(filter.apply(['d', 'a', 'b', 'c'])).toEqual(['b', 'a']);
  expect(filter.apply(['d', 'a', 'c'])).toEqual(['a']);
  expect(filter.apply(['b'])).toEqual(['b']);
  expect(filter.apply(['c', 'd'])).toEqual([]);
});

test('no filter', () => {
  // This filter should just return the input keys in sorted order
  const filter = new KeyFilter('');
  expect(filter.apply(['d', 'a', 'b', 'c'])).toEqual(['a', 'b', 'c', 'd']);
  expect(filter.apply(['d', 'a', 'c'])).toEqual(['a', 'c', 'd']);
  expect(filter.apply([])).toEqual([]);
});

test('no constructor argument given', () => {
  const filter = new KeyFilter();
  expect(filter.apply(['d', 'a', 'b', 'c'])).toEqual(['a', 'b', 'c', 'd']);
  expect(filter.apply(['d', 'a', 'c'])).toEqual(['a', 'c', 'd']);
  expect(filter.apply([])).toEqual([]);
});

test('spaces and empty strings in parsing', () => {
  expect(new KeyFilter('').keyList).toEqual([]);
  expect(new KeyFilter(' ').keyList).toEqual([]);
  expect(new KeyFilter(' , \t,').keyList).toEqual([]);
  expect(new KeyFilter('a').keyList).toEqual(['a']);
  expect(new KeyFilter('   a \t').keyList).toEqual(['a']);
  expect(new KeyFilter('b, a').keyList).toEqual(['b', 'a']);
  expect(new KeyFilter(' b,  , a ').keyList).toEqual(['b', 'a']);
});

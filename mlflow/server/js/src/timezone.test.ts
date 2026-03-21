import { test, expect } from '@jest/globals';

test('timezone is GMT', () => {
  const d = new Date();
  expect(d.getTimezoneOffset()).toBe(0);
});

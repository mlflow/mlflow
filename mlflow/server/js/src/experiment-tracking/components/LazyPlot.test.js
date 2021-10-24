import React from 'react';
import { LazyPlot, retry } from './LazyPlot';
import { shallowWithIntl } from '../../common/utils/TestUtils';

describe('retry', () => {
  let fn;

  it('calls the function', async () => {
    fn = jest.fn();
    await retry(fn);
    expect(fn).toHaveBeenCalled();
  });

  it('yields the correct value value from the function', async () => {
    const value = 'Doh!';
    fn = () => Promise.resolve(value);
    const returnedValue = await retry(fn);
    expect(returnedValue).toEqual(value);
  });

  it('retries the function n times before throwing errors', async () => {
    const errorMessage = 'I am a failure';
    fn = jest.fn(() => Promise.reject(new Error(errorMessage)));
    const n = 2;

    await expect(retry(fn, 2, 10)).rejects.toThrow(errorMessage);
    expect(fn).toHaveBeenCalledTimes(n + 1);
  });
});

describe('LazyPlot', () => {
  beforeEach(() => {
    jest.mock('react-plotly.js', () => <div />);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should render with minimal props without exploding', () => {
    const wrapper = shallowWithIntl(<LazyPlot />);
    expect(wrapper.length).toBe(1);
  });
});

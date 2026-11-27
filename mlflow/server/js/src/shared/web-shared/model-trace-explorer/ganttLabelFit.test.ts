import { afterEach, describe, expect, it, jest } from '@jest/globals';

import { fitGanttLabel } from './ganttLabelFit';

const label = (width: number) => {
  const element = document.createElement('span');
  element.style.display = 'none';
  const measure = jest.fn(() => width);
  Object.defineProperty(element, 'offsetWidth', { get: measure });
  document.body.appendChild(element);
  return { element, measure };
};

afterEach(() => document.body.replaceChildren());

describe('fitGanttLabel', () => {
  it('uses the latest width for repeated requests and measures once', async () => {
    const inside = label(40);
    const beside = label(50);
    fitGanttLabel(inside.element, beside.element, 100);
    fitGanttLabel(inside.element, beside.element, 50);
    await Promise.resolve();
    expect(inside.measure).toHaveBeenCalledTimes(1);
    expect(beside.measure).toHaveBeenCalledTimes(1);
    expect([inside.element.style.display, beside.element.style.display]).toEqual(['none', 'inline-flex']);
    fitGanttLabel(inside.element, beside.element, 51);
    await Promise.resolve();
    expect([inside.element.style.display, beside.element.style.display]).toEqual(['inline-flex', 'none']);
  });

  it.each(['inside', 'beside'])('skips a pair whose %s label unmounted', async (removed) => {
    const inside = label(40);
    const beside = label(40);
    fitGanttLabel(inside.element, beside.element, 100);
    (removed === 'inside' ? inside : beside).element.remove();
    await Promise.resolve();
    expect(inside.measure).not.toHaveBeenCalled();
    expect(beside.measure).not.toHaveBeenCalled();
    expect([inside.element.style.display, beside.element.style.display]).toEqual(['none', 'none']);
  });
});

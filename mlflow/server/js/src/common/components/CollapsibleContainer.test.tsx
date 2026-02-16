import { it, describe, jest, expect } from '@jest/globals';
import React from 'react';

import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import { CollapsibleContainer } from './CollapsibleContainer';

// Mock ResizeObserver
const mockResizeObserver: any = jest.fn();
mockResizeObserver.mockImplementation((callback: any) => ({
  observe: jest.fn(() => callback([{ target: {} }])),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

global.ResizeObserver = mockResizeObserver;

describe('CollapsibleContainer', () => {
  it('renders children correctly', () => {
    const setIsExpanded = jest.fn();
    renderWithIntl(
      <CollapsibleContainer isExpanded={false} setIsExpanded={setIsExpanded}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );
    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('renders container with proper structure', () => {
    const setIsExpanded = jest.fn();
    renderWithIntl(
      <CollapsibleContainer isExpanded={false} setIsExpanded={setIsExpanded} maxHeight={100}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );
    const container = screen.getByText('Test content').parentElement;
    expect(container).toBeInTheDocument();
    expect(container).toContainHTML('<div>Test content</div>');
  });

  it('shows gradient when collapsed and content is taller than maxHeight', () => {
    const setIsExpanded = jest.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, value: 200 });

    renderWithIntl(
      <CollapsibleContainer isExpanded={false} setIsExpanded={setIsExpanded} maxHeight={100}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );

    expect(screen.getByTestId('truncation-gradient')).toBeInTheDocument();
  });

  it('does not show gradient when expanded', () => {
    const setIsExpanded = jest.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, value: 200 });

    renderWithIntl(
      <CollapsibleContainer isExpanded setIsExpanded={setIsExpanded} maxHeight={100}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );

    expect(screen.queryByTestId('truncation-gradient')).not.toBeInTheDocument();
  });

  it('shows "Show more" button when content is collapsible', () => {
    const setIsExpanded = jest.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, value: 200 });

    renderWithIntl(
      <CollapsibleContainer isExpanded={false} setIsExpanded={setIsExpanded} maxHeight={100}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );

    expect(screen.getByText('Show more')).toBeInTheDocument();
  });

  it('shows "Show less" button when expanded', () => {
    const setIsExpanded = jest.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, value: 200 });

    renderWithIntl(
      <CollapsibleContainer isExpanded setIsExpanded={setIsExpanded} maxHeight={100}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );

    expect(screen.getByText('Show less')).toBeInTheDocument();
  });

  it('calls setIsExpanded with a function when toggle button is clicked', async () => {
    const setIsExpanded: any = jest.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollHeight', { configurable: true, value: 200 });

    renderWithIntl(
      <CollapsibleContainer isExpanded={false} setIsExpanded={setIsExpanded} maxHeight={100}>
        <div>Test content</div>
      </CollapsibleContainer>,
    );

    await userEvent.click(screen.getByText('Show more'));
    expect(setIsExpanded).toHaveBeenCalledTimes(1);
    expect(setIsExpanded).toHaveBeenCalledWith(expect.any(Function));

    // Call the function passed to setIsExpanded and check its result
    const updateFunction = setIsExpanded.mock.calls[0][0];
    expect(updateFunction(false)).toBe(true);
    expect(updateFunction(true)).toBe(false);
  });
});

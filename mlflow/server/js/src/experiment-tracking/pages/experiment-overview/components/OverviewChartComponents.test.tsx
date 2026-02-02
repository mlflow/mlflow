import { describe, it, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { useScrollableLegendProps } from './OverviewChartComponents';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>{children}</DesignSystemProvider>
);

describe('useScrollableLegendProps', () => {
  it('should return formatter and wrapperStyle', () => {
    const { result } = renderHook(() => useScrollableLegendProps(), { wrapper });

    expect(result.current).toHaveProperty('formatter');
    expect(result.current).toHaveProperty('wrapperStyle');
    expect(typeof result.current.formatter).toBe('function');
    expect(typeof result.current.wrapperStyle).toBe('object');
  });

  it('should return wrapperStyle with default maxHeight of 60', () => {
    const { result } = renderHook(() => useScrollableLegendProps(), { wrapper });

    expect(result.current.wrapperStyle.maxHeight).toBe(60);
    expect(result.current.wrapperStyle.overflowY).toBe('auto');
    expect(result.current.wrapperStyle.overflowX).toBe('hidden');
  });

  it('should allow custom maxHeight via config', () => {
    const { result } = renderHook(() => useScrollableLegendProps({ maxHeight: 100 }), { wrapper });

    expect(result.current.wrapperStyle.maxHeight).toBe(100);
  });

  it('should return a formatter that renders a span with the value', () => {
    const { result } = renderHook(() => useScrollableLegendProps(), { wrapper });

    const formattedElement = result.current.formatter('Test Legend');
    expect(formattedElement).toBeTruthy();
    expect(formattedElement.type).toBe('span');
    expect(formattedElement.props.children).toBe('Test Legend');
  });

  it('should apply correct styles to formatted legend text', () => {
    const { result } = renderHook(() => useScrollableLegendProps(), { wrapper });

    const formattedElement = result.current.formatter('Test');
    // The component uses Emotion's css prop, not style
    expect(formattedElement.props.css).toHaveProperty('cursor', 'pointer');
    expect(formattedElement.props.css).toHaveProperty('color');
    expect(formattedElement.props.css).toHaveProperty('fontSize');
  });
});

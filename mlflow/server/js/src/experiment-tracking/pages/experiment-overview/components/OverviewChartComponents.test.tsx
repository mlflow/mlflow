import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import {
  useScrollableLegendProps,
  getTracesFilteredByTimeRangeUrl,
  getTracesFilteredUrl,
  createAssessmentEqualsFilter,
  createAssessmentExistsFilter,
  createSpanNameEqualsFilter,
  createSpanStatusEqualsFilter,
  ScrollableTooltip,
} from './OverviewChartComponents';
import { OverviewChartContext, type OverviewChartContextValue } from '../OverviewChartContext';
import { MemoryRouter, useLocation } from '../../../../common/utils/RoutingUtils';
import { IntlProvider } from 'react-intl';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>{children}</DesignSystemProvider>
);

const renderTooltipWithProviders = (props: React.ComponentProps<typeof ScrollableTooltip>) => {
  return render(
    <MemoryRouter>
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ScrollableTooltip {...props} />
        </DesignSystemProvider>
      </IntlProvider>
    </MemoryRouter>,
  );
};

// Surfaces the current router location as text so tests can assert post-click navigation
// without mocking useNavigate.
const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>;
};

const renderTooltipWithContext = (
  props: React.ComponentProps<typeof ScrollableTooltip>,
  contextValue: Partial<OverviewChartContextValue>,
) => {
  const value: OverviewChartContextValue = {
    experimentIds: ['exp-1'],
    timeIntervalSeconds: 3600,
    timeBuckets: [],
    ...contextValue,
  };
  return render(
    <MemoryRouter>
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <OverviewChartContext.Provider value={value}>
            <ScrollableTooltip {...props} />
            <LocationProbe />
          </OverviewChartContext.Provider>
        </DesignSystemProvider>
      </IntlProvider>
    </MemoryRouter>,
  );
};

describe('getTracesFilteredByTimeRangeUrl', () => {
  it('should generate correct URL with time range parameters', () => {
    const experimentId = 'test-experiment-123';
    const timestampMs = new Date('2025-12-22T10:00:00Z').getTime();
    const timeIntervalSeconds = 3600; // 1 hour

    const url = getTracesFilteredByTimeRangeUrl(experimentId, timestampMs, timeIntervalSeconds);

    expect(url).toContain('/experiments/test-experiment-123/traces');
    expect(url).toContain('startTimeLabel=CUSTOM');
    expect(url).toContain('startTime=2025-12-22T10%3A00%3A00.000Z');
    expect(url).toContain('endTime=2025-12-22T11%3A00%3A00.000Z');
  });

  it('should calculate end time based on time interval', () => {
    const experimentId = 'exp-1';
    const timestampMs = new Date('2025-01-01T00:00:00Z').getTime();

    // Test with 1 day interval
    const urlDaily = getTracesFilteredByTimeRangeUrl(experimentId, timestampMs, 86400);
    expect(urlDaily).toContain('endTime=2025-01-02T00%3A00%3A00.000Z');

    // Test with 1 minute interval
    const urlMinute = getTracesFilteredByTimeRangeUrl(experimentId, timestampMs, 60);
    expect(urlMinute).toContain('endTime=2025-01-01T00%3A01%3A00.000Z');
  });

  it('should include all required query parameters', () => {
    const url = getTracesFilteredByTimeRangeUrl('exp-1', Date.now(), 3600);

    expect(url).toContain('startTimeLabel=CUSTOM');
    expect(url).toContain('startTime=');
    expect(url).toContain('endTime=');
  });
});

describe('getTracesFilteredUrl', () => {
  it('should generate correct URL with time range and filters', () => {
    const experimentId = 'test-experiment-123';
    const timeRange = {
      startTimeLabel: 'CUSTOM',
      startTime: '2025-01-01T00:00:00Z',
      endTime: '2025-01-02T00:00:00Z',
    };

    const url = getTracesFilteredUrl(experimentId, timeRange, [createAssessmentEqualsFilter('quality', 'good')]);

    expect(url).toContain('/experiments/test-experiment-123/traces');
    expect(url).toContain('startTimeLabel=CUSTOM');
    expect(url).toContain('startTime=2025-01-01T00%3A00%3A00Z');
    expect(url).toContain('endTime=2025-01-02T00%3A00%3A00Z');
    expect(url).toContain('filter=ASSESSMENT%3A%3A%3D%3A%3Agood%3A%3Aquality');
  });

  it('should generate URL without time range when not provided', () => {
    const url = getTracesFilteredUrl('exp-1', undefined, [createAssessmentEqualsFilter('accuracy', '0.95')]);

    expect(url).toContain('/experiments/exp-1/traces');
    expect(url).not.toContain('startTimeLabel');
    expect(url).toContain('filter=ASSESSMENT%3A%3A%3D%3A%3A0.95%3A%3Aaccuracy');
  });

  it('should generate URL without filters when not provided', () => {
    const url = getTracesFilteredUrl('exp-1', { startTimeLabel: 'LAST_7_DAYS' });

    expect(url).toContain('/experiments/exp-1/traces');
    expect(url).toContain('startTimeLabel=LAST_7_DAYS');
    expect(url).not.toContain('filter=');
  });
});

describe('createAssessmentEqualsFilter', () => {
  it('should create correct filter string format', () => {
    const filter = createAssessmentEqualsFilter('quality', 'good');
    expect(filter).toBe('ASSESSMENT::=::good::quality');
  });
});

describe('createAssessmentExistsFilter', () => {
  it('should create correct filter string format', () => {
    const filter = createAssessmentExistsFilter('quality');
    expect(filter).toBe('ASSESSMENT::IS NOT NULL::::quality');
  });
});

describe('createSpanNameEqualsFilter', () => {
  it('should create correct filter string format', () => {
    const filter = createSpanNameEqualsFilter('my_tool');
    expect(filter).toBe('span.name::=::my_tool');
  });

  it('should handle tool names with special characters', () => {
    const filter = createSpanNameEqualsFilter('get_weather_data');
    expect(filter).toBe('span.name::=::get_weather_data');
  });
});

describe('createSpanStatusEqualsFilter', () => {
  it('should create correct filter string for ERROR status', () => {
    const filter = createSpanStatusEqualsFilter('ERROR');
    expect(filter).toBe('span.status::=::ERROR');
  });

  it('should create correct filter string for OK status', () => {
    const filter = createSpanStatusEqualsFilter('OK');
    expect(filter).toBe('span.status::=::OK');
  });
});

describe('ScrollableTooltip', () => {
  const mockFormatter = (value: number, name: string): [string, string] => [`${value}`, name];

  it('should not render when not active', () => {
    renderTooltipWithProviders({
      active: false,
      payload: [{ payload: { timestampMs: 1234567890 }, name: 'count', value: 42, color: 'blue' }],
      label: 'Test Label',
      formatter: mockFormatter,
      componentId: 'mlflow.overview.usage.traces.view_traces_link',
    });

    expect(screen.queryByText('Test Label')).not.toBeInTheDocument();
  });

  it('should render tooltip content when active', () => {
    renderTooltipWithProviders({
      active: true,
      payload: [{ payload: { timestampMs: 1234567890 }, name: 'count', value: 42, color: 'blue' }],
      label: 'Test Label',
      formatter: mockFormatter,
      componentId: 'mlflow.overview.usage.traces.view_traces_link',
    });

    expect(screen.getByText('Test Label')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('count:')).toBeInTheDocument();
  });

  it('should not show link when linkConfig is not provided', () => {
    renderTooltipWithProviders({
      active: true,
      payload: [{ payload: { timestampMs: 1234567890 }, name: 'count', value: 42, color: 'blue' }],
      label: 'Test Label',
      formatter: mockFormatter,
      componentId: 'mlflow.overview.usage.traces.view_traces_link',
    });

    expect(screen.queryByText('View traces for this period')).not.toBeInTheDocument();
  });

  it('should show "View traces for this period" link when linkConfig is provided', () => {
    renderTooltipWithProviders({
      active: true,
      payload: [{ payload: { timestampMs: 1234567890 }, name: 'count', value: 42, color: 'blue' }],
      label: 'Test Label',
      formatter: mockFormatter,
      linkConfig: {
        experimentId: 'test-exp-123',
        timeIntervalSeconds: 3600,
      },
      componentId: 'mlflow.overview.usage.traces.view_traces_link',
    });

    expect(screen.getByText('View traces for this period')).toBeInTheDocument();
  });

  it('should not show link when payload has no timestampMs', () => {
    renderTooltipWithProviders({
      active: true,
      payload: [{ payload: {}, name: 'count', value: 42, color: 'blue' }],
      label: 'Test Label',
      formatter: mockFormatter,
      linkConfig: {
        experimentId: 'test-exp-123',
        timeIntervalSeconds: 3600,
      },
      componentId: 'mlflow.overview.usage.traces.view_traces_link',
    });

    expect(screen.queryByText('View traces for this period')).not.toBeInTheDocument();
  });

  describe('navigation forwards tracesNavigationFilters from context', () => {
    const timestampMs = new Date('2025-12-22T10:00:00Z').getTime();

    it('appends tracesNavigationFilters from context as filter query params', async () => {
      const user = userEvent.setup();
      renderTooltipWithContext(
        {
          active: true,
          payload: [{ payload: { timestampMs }, name: 'count', value: 42, color: 'blue' }],
          label: 'Test Label',
          formatter: mockFormatter,
          linkConfig: {
            experimentId: 'test-exp-123',
            timeIntervalSeconds: 3600,
          },
          componentId: 'mlflow.overview.usage.traces.view_traces_link',
        },
        { tracesNavigationFilters: ['user::=::bob'] },
      );

      await user.click(screen.getByText('View traces for this period'));

      const location = screen.getByTestId('location-probe').textContent ?? '';
      expect(location).toContain('/experiments/test-exp-123/traces');
      expect(location).toContain('startTimeLabel=CUSTOM');
      expect(location).toContain('startTime=2025-12-22T10%3A00%3A00.000Z');
      expect(location).toContain('endTime=2025-12-22T11%3A00%3A00.000Z');
      expect(location).toContain('filter=user%3A%3A%3D%3A%3Abob');
    });

    it('appends every entry when multiple tracesNavigationFilters are provided', async () => {
      const user = userEvent.setup();
      renderTooltipWithContext(
        {
          active: true,
          payload: [{ payload: { timestampMs }, name: 'count', value: 42, color: 'blue' }],
          label: 'Test Label',
          formatter: mockFormatter,
          linkConfig: {
            experimentId: 'test-exp-123',
            timeIntervalSeconds: 3600,
          },
          componentId: 'mlflow.overview.usage.traces.view_traces_link',
        },
        { tracesNavigationFilters: ['user::=::bob', 'user::=::alice'] },
      );

      await user.click(screen.getByText('View traces for this period'));

      const location = screen.getByTestId('location-probe').textContent ?? '';
      expect(location).toContain('filter=user%3A%3A%3D%3A%3Abob');
      expect(location).toContain('filter=user%3A%3A%3D%3A%3Aalice');
    });

    it('omits filter query param when tracesNavigationFilters is undefined', async () => {
      const user = userEvent.setup();
      renderTooltipWithContext(
        {
          active: true,
          payload: [{ payload: { timestampMs }, name: 'count', value: 42, color: 'blue' }],
          label: 'Test Label',
          formatter: mockFormatter,
          linkConfig: {
            experimentId: 'test-exp-123',
            timeIntervalSeconds: 3600,
          },
          componentId: 'mlflow.overview.usage.traces.view_traces_link',
        },
        {},
      );

      await user.click(screen.getByText('View traces for this period'));

      const location = screen.getByTestId('location-probe').textContent ?? '';
      expect(location).toContain('/experiments/test-exp-123/traces');
      expect(location).toContain('startTimeLabel=CUSTOM');
      expect(location).not.toContain('filter=');
    });

    it('does not forward tracesNavigationFilters when a custom onLinkClick is used', async () => {
      const user = userEvent.setup();
      const onLinkClick = jest.fn();
      renderTooltipWithContext(
        {
          active: true,
          payload: [{ payload: { timestampMs }, name: 'count', value: 42, color: 'blue' }],
          label: 'Test Label',
          formatter: mockFormatter,
          linkConfig: { onLinkClick },
          componentId: 'mlflow.overview.usage.errors.view_traces_link',
        },
        { tracesNavigationFilters: ['user::=::bob'] },
      );

      await user.click(screen.getByText('View traces for this period'));

      // Custom onLinkClick path is opaque to ScrollableTooltip; the chart's own handler is
      // responsible for merging tracesNavigationFilters into its URL (covered by chart-level tests).
      expect(onLinkClick).toHaveBeenCalledTimes(1);
      const location = screen.getByTestId('location-probe').textContent ?? '';
      expect(location).not.toContain('filter=');
    });
  });
});

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

import { jest, describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useOverviewTab, OverviewTab } from './useOverviewTab';

// Mock the routing utils
const mockNavigate = jest.fn();
const mockUseParams = jest.fn();
const mockUseLocation = jest.fn();

jest.mock('@mlflow/mlflow/src/common/utils/RoutingUtils', () => ({
  useNavigate: () => mockNavigate,
  useParams: () => mockUseParams(),
  useLocation: () => mockUseLocation(),
}));

describe('useOverviewTab', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default mock values
    mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: undefined });
    mockUseLocation.mockReturnValue({ search: '' });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('activeTab', () => {
    it('should return "usage" as default when no tab in URL', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: undefined });

      const { result } = renderHook(() => useOverviewTab());

      expect(result.current[0]).toBe(OverviewTab.Usage);
    });

    it('should return "usage" when overviewTab is "usage"', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: 'usage' });

      const { result } = renderHook(() => useOverviewTab());

      expect(result.current[0]).toBe(OverviewTab.Usage);
    });

    it('should return "quality" when overviewTab is "quality"', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: 'quality' });

      const { result } = renderHook(() => useOverviewTab());

      expect(result.current[0]).toBe(OverviewTab.Quality);
    });

    it('should return "tool-calls" when overviewTab is "tool-calls"', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: 'tool-calls' });

      const { result } = renderHook(() => useOverviewTab());

      expect(result.current[0]).toBe(OverviewTab.ToolCalls);
    });

    it('should return default "usage" when overviewTab is invalid', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: 'invalid-tab' });

      const { result } = renderHook(() => useOverviewTab());

      expect(result.current[0]).toBe(OverviewTab.Usage);
    });
  });

  describe('setActiveTab', () => {
    it('should navigate to the new tab', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: 'usage' });
      mockUseLocation.mockReturnValue({ search: '' });

      const { result } = renderHook(() => useOverviewTab());

      act(() => {
        result.current[1](OverviewTab.Quality);
      });

      expect(mockNavigate).toHaveBeenCalledWith('/experiments/123/overview/quality', { replace: true });
    });

    it('should preserve query params when changing tabs', () => {
      mockUseParams.mockReturnValue({ experimentId: '123', overviewTab: 'usage' });
      mockUseLocation.mockReturnValue({ search: '?startTimeLabel=LAST_7_DAYS&foo=bar' });

      const { result } = renderHook(() => useOverviewTab());

      act(() => {
        result.current[1](OverviewTab.ToolCalls);
      });

      expect(mockNavigate).toHaveBeenCalledWith(
        '/experiments/123/overview/tool-calls?startTimeLabel=LAST_7_DAYS&foo=bar',
        { replace: true },
      );
    });

    it('should navigate to usage tab', () => {
      mockUseParams.mockReturnValue({ experimentId: '456', overviewTab: 'quality' });
      mockUseLocation.mockReturnValue({ search: '' });

      const { result } = renderHook(() => useOverviewTab());

      act(() => {
        result.current[1](OverviewTab.Usage);
      });

      expect(mockNavigate).toHaveBeenCalledWith('/experiments/456/overview/usage', { replace: true });
    });
  });

  describe('OverviewTab enum', () => {
    it('should have correct values', () => {
      expect(OverviewTab.Usage).toBe('usage');
      expect(OverviewTab.Quality).toBe('quality');
      expect(OverviewTab.ToolCalls).toBe('tool-calls');
    });
  });
});

import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useDocumentTitle } from './useDocumentTitle';
import * as RoutingUtils from '../utils/RoutingUtils';

// Mock the useMatches hook from RoutingUtils
jest.mock('../utils/RoutingUtils', () => ({
  useMatches: jest.fn(),
}));

describe('useDocumentTitle', () => {
  const mockUseMatches = RoutingUtils.useMatches as jest.MockedFunction<typeof RoutingUtils.useMatches>;

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset document title before each test
    document.title = '';
  });

  it('sets document title from route handle getPageTitle function', () => {
    // Mock useMatches to return a route with a getPageTitle function (after processRouteDefs)
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'Test Page' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Test Page - MLflow');
  });

  it('passes route params to getPageTitle function', () => {
    // Mock useMatches to return a route with params
    mockUseMatches.mockReturnValue([
      {
        id: 'experiment',
        pathname: '/experiments/123',
        params: { experimentId: '123' },
        data: undefined,
        handle: { getPageTitle: (params: Record<string, string | undefined>) => `Experiment ${params['experimentId']}` },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Experiment 123 - MLflow');
  });

  it('uses the last matching route title when multiple routes have getPageTitle', () => {
    // Mock useMatches to return multiple routes with getPageTitle functions
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'Root' },
      },
      {
        id: 'parent',
        pathname: '/parent',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'Parent' },
      },
      {
        id: 'child',
        pathname: '/parent/child',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'Child' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    // Should use the most specific (last) route title
    expect(document.title).toBe('Child - MLflow');
  });

  it('falls back to "MLflow" when no route has a getPageTitle function', () => {
    // Mock useMatches to return routes without getPageTitle
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: undefined,
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('MLflow');
  });

  it('falls back to "MLflow" when routes have handles but no getPageTitle function', () => {
    // Mock useMatches to return routes with handles but no getPageTitle
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: { someOtherProperty: 'value' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('MLflow');
  });

  it('updates document title when route changes', () => {
    // Start with one title
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'First Page' },
      },
    ]);

    const { rerender } = renderHook(() => useDocumentTitle());
    expect(document.title).toBe('First Page - MLflow');

    // Change to a different title
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/other',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'Second Page' },
      },
    ]);

    rerender();
    expect(document.title).toBe('Second Page - MLflow');
  });

  it('handles AI Gateway routes correctly', () => {
    mockUseMatches.mockReturnValue([
      {
        id: 'gateway',
        pathname: '/gateway',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'AI Gateway' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('AI Gateway - MLflow');
  });

  it('handles nested gateway routes correctly', () => {
    mockUseMatches.mockReturnValue([
      {
        id: 'gateway',
        pathname: '/gateway',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'AI Gateway' },
      },
      {
        id: 'api-keys',
        pathname: '/gateway/api-keys',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'API Keys' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('API Keys - MLflow');
  });

  it('handles endpoint details page with params correctly', () => {
    mockUseMatches.mockReturnValue([
      {
        id: 'gateway',
        pathname: '/gateway',
        params: {},
        data: undefined,
        handle: { getPageTitle: () => 'AI Gateway' },
      },
      {
        id: 'endpoint-details',
        pathname: '/gateway/endpoints/test-123',
        params: { endpointId: 'test-123' },
        data: undefined,
        handle: { getPageTitle: (params: Record<string, string | undefined>) => `Endpoint ${params['endpointId']}` },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Endpoint test-123 - MLflow');
  });

  it('handles experiment page with experimentId param', () => {
    mockUseMatches.mockReturnValue([
      {
        id: 'experiment',
        pathname: '/experiments/456',
        params: { experimentId: '456' },
        data: undefined,
        handle: { getPageTitle: (params: Record<string, string | undefined>) => `Experiment ${params['experimentId']}` },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Experiment 456 - MLflow');
  });

  it('handles run page with multiple params', () => {
    mockUseMatches.mockReturnValue([
      {
        id: 'run',
        pathname: '/experiments/123/runs/abc-def',
        params: { experimentId: '123', runUuid: 'abc-def' },
        data: undefined,
        handle: { getPageTitle: (params: Record<string, string | undefined>) => `Run ${params['runUuid']}` },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Run abc-def - MLflow');
  });
});

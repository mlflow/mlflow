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

  it('sets document title from route handle', () => {
    // Mock useMatches to return a route with a title
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: { title: 'Test Page' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Test Page - MLflow');
  });

  it('uses the last matching route title when multiple routes have titles', () => {
    // Mock useMatches to return multiple routes with titles
    mockUseMatches.mockReturnValue([
      {
        id: 'root',
        pathname: '/',
        params: {},
        data: undefined,
        handle: { title: 'Root' },
      },
      {
        id: 'parent',
        pathname: '/parent',
        params: {},
        data: undefined,
        handle: { title: 'Parent' },
      },
      {
        id: 'child',
        pathname: '/parent/child',
        params: {},
        data: undefined,
        handle: { title: 'Child' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    // Should use the most specific (last) route title
    expect(document.title).toBe('Child - MLflow');
  });

  it('falls back to "MLflow" when no route has a title', () => {
    // Mock useMatches to return routes without titles
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

  it('falls back to "MLflow" when routes have handles but no title property', () => {
    // Mock useMatches to return routes with handles but no title
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
        handle: { title: 'First Page' },
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
        handle: { title: 'Second Page' },
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
        handle: { title: 'AI Gateway' },
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
        handle: { title: 'AI Gateway' },
      },
      {
        id: 'api-keys',
        pathname: '/gateway/api-keys',
        params: {},
        data: undefined,
        handle: { title: 'API Keys' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('API Keys - MLflow');
  });

  it('handles endpoint details page correctly', () => {
    mockUseMatches.mockReturnValue([
      {
        id: 'gateway',
        pathname: '/gateway',
        params: {},
        data: undefined,
        handle: { title: 'AI Gateway' },
      },
      {
        id: 'endpoint-details',
        pathname: '/gateway/endpoints/test-123',
        params: { endpointId: 'test-123' },
        data: undefined,
        handle: { title: 'Endpoint Details' },
      },
    ]);

    renderHook(() => useDocumentTitle());

    expect(document.title).toBe('Endpoint Details - MLflow');
  });
});

import { describe, test, expect } from '@jest/globals';
import { getGatewayRouteDefs } from './route-defs';

describe('Gateway Route Definitions', () => {
  test('AI Gateway page has correct title in handle', () => {
    const routes = getGatewayRouteDefs();
    const gatewayRoute = routes.find((route) => route.path === '/gateway');

    expect(gatewayRoute).toBeDefined();
    expect(gatewayRoute?.handle).toEqual({ title: 'AI Gateway' });
  });

  test('API Keys page has correct title in handle', () => {
    const routes = getGatewayRouteDefs();
    const gatewayRoute = routes.find((route) => route.path === '/gateway');
    const apiKeysRoute = gatewayRoute?.children?.find((child) => child.path === 'api-keys');

    expect(apiKeysRoute).toBeDefined();
    expect(apiKeysRoute?.handle).toEqual({ title: 'API Keys' });
  });

  test('Create Endpoint page has correct title in handle', () => {
    const routes = getGatewayRouteDefs();
    const gatewayRoute = routes.find((route) => route.path === '/gateway');
    const createRoute = gatewayRoute?.children?.find((child) => child.path === 'endpoints/create');

    expect(createRoute).toBeDefined();
    expect(createRoute?.handle).toEqual({ title: 'Create Endpoint' });
  });

  test('Endpoint Details page has correct title in handle', () => {
    const routes = getGatewayRouteDefs();
    const gatewayRoute = routes.find((route) => route.path === '/gateway');
    const detailsRoute = gatewayRoute?.children?.find(
      (child) => child.path === 'endpoints/:endpointId',
    );

    expect(detailsRoute).toBeDefined();
    expect(detailsRoute?.handle).toEqual({ title: 'Endpoint Details' });
  });

  test('route structure does not include separate edit route', () => {
    const routes = getGatewayRouteDefs();

    // Find the gateway page route
    const gatewayRoute = routes.find((route) => route.path === '/gateway');
    expect(gatewayRoute).toBeDefined();

    // Check that there's no separate /edit route
    const editRoute = gatewayRoute?.children?.find(
      (child) => child.path === 'endpoints/:endpointId/edit',
    );
    expect(editRoute).toBeUndefined();

    // Verify the endpoint details route exists
    const detailsRoute = gatewayRoute?.children?.find(
      (child) => child.path === 'endpoints/:endpointId',
    );
    expect(detailsRoute).toBeDefined();
    expect(detailsRoute?.handle).toEqual({ title: 'Endpoint Details' });
  });
});

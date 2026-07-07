import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { useLegacySelectedDatasetRedirect } from './useLegacySelectedDatasetRedirect';

// OSS's `setupTestRouter` is a no-op stub (`{ history: {} }`) — it provides nothing
// useful to assert against. We use react-router's `MemoryRouter` directly and capture
// the current location via a tiny probe component, exposing it through a closure.

const Probe = () => {
  const { isRedirecting } = useLegacySelectedDatasetRedirect();
  return <div data-testid="probe">{isRedirecting ? 'redirecting' : 'idle'}</div>;
};

const TargetStub = () => <div data-testid="detail-target">detail</div>;

interface LocationSnapshot {
  pathname: string;
  search: string;
}

const LocationProbe = ({ onLocation }: { onLocation: (loc: LocationSnapshot) => void }) => {
  const location = useLocation();
  onLocation({ pathname: location.pathname, search: location.search });
  return null;
};

const renderWithRouter = (initialUrl: string) => {
  let currentLocation: LocationSnapshot = { pathname: '', search: '' };
  const result = render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <LocationProbe onLocation={(loc) => (currentLocation = loc)} />
      <Routes>
        <Route path="/experiments/:experimentId/datasets" element={<Probe />} />
        <Route path="/experiments/:experimentId/datasets/:datasetId" element={<TargetStub />} />
      </Routes>
    </MemoryRouter>,
  );
  return { ...result, getLocation: () => currentLocation };
};

describe('useLegacySelectedDatasetRedirect', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('rewrites a legacy `?selectedDatasetId=…` URL to the V2 detail route', async () => {
    const { getLocation } = renderWithRouter('/experiments/exp-1/datasets?selectedDatasetId=ds-42');
    await waitFor(() => {
      expect(getLocation().pathname).toContain('/experiments/exp-1/datasets/ds-42');
    });
    expect(getLocation().search).not.toContain('selectedDatasetId');
  });

  test('preserves unrelated query params when redirecting', async () => {
    const { getLocation } = renderWithRouter(
      '/experiments/exp-1/datasets?selectedDatasetId=ds-42&keep=me&another=value',
    );
    await waitFor(() => {
      expect(getLocation().pathname).toContain('/experiments/exp-1/datasets/ds-42');
    });
    expect(getLocation().search).toContain('keep=me');
    expect(getLocation().search).toContain('another=value');
    expect(getLocation().search).not.toContain('selectedDatasetId');
  });

  test('is a no-op when the legacy param is absent', async () => {
    const { getLocation } = renderWithRouter('/experiments/exp-1/datasets');
    const probe = await screen.findByTestId('probe');
    expect(probe.textContent).toBe('idle');
    expect(getLocation().pathname).toMatch(/\/experiments\/exp-1\/datasets$/);
  });
});

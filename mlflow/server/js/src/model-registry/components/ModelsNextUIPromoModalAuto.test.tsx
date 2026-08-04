import { jest, describe, test, expect } from '@jest/globals';
import { renderWithIntl, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { withNextModelsUIContext } from '../hooks/useNextModelsUI';
import { ModelsNextUIPromoModalAuto } from './ModelsNextUIPromoModalAuto';

jest.mock('../../common/utils/FeatureUtils', () => ({
  shouldShowModelsNextUI: () => true,
}));

describe('ModelsNextUIPromoModalAuto', () => {
  const renderComponent = () => {
    const Component = withNextModelsUIContext(() => <ModelsNextUIPromoModalAuto />);
    return renderWithIntl(<Component />);
  };

  const mockSeenPromoModal = () =>
    jest.spyOn(window.localStorage, 'getItem').mockImplementation((key) => (key.match(/promo/) ? 'true' : ''));

  const mockUnseenPromoModal = () =>
    jest.spyOn(window.localStorage, 'getItem').mockImplementation((key) => (key.match(/promo/) ? 'false' : ''));

  test('it displays the promo modal when not previously seen (independent of any models)', () => {
    mockUnseenPromoModal();
    renderComponent();
    expect(within(screen.getByRole('dialog')).getByText(/Flexible, governed deployments/)).toBeInTheDocument();
  });

  test('it does not display the promo modal when already seen', () => {
    mockSeenPromoModal();
    renderComponent();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

import userEvent from '@testing-library/user-event';
import { renderWithIntl, act, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { useNextModelsUIContext, withNextModelsUIContext } from '../hooks/useNextModelsUI';
import { ModelsNextUIToggleSwitch } from './ModelsNextUIToggleSwitch';

jest.mock('../../common/utils/FeatureUtils', () => ({
  shouldShowModelsNextUI: () => true,
}));

describe('ModelsNextUIToggleSwitch', () => {
  const renderTestComponent = (additionalElement: React.ReactNode = null) => {
    const Component = withNextModelsUIContext(() => {
      return (
        <>
          <ModelsNextUIToggleSwitch />
          {additionalElement}
        </>
      );
    });

    return renderWithIntl(<Component />);
  };

  const mockSeenPromoModal = () =>
    jest.spyOn(window.localStorage, 'getItem').mockImplementation((key) => (key.match(/promo/) ? 'true' : ''));

  const mockUnseenPromoModal = () =>
    jest.spyOn(window.localStorage, 'getItem').mockImplementation((key) => (key.match(/promo/) ? 'false' : ''));

  test('it should render the switch and display the promo modal', () => {
    mockUnseenPromoModal();

    renderTestComponent();
    expect(screen.getByRole('switch', { name: /New model registry UI/ })).toBeInTheDocument();
    expect(within(screen.getByRole('dialog')).getByText(/Flexible, governed deployments/)).toBeInTheDocument();
  });

  test("it should not render display the promo modal when it's already seen", () => {
    mockSeenPromoModal();
    renderTestComponent();
    expect(screen.getByRole('switch')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('it should display confirmation modal when trying to switch off', async () => {
    mockSeenPromoModal();

    const PreviewComponent = () => {
      const { usingNextModelsUI } = useNextModelsUIContext();
      return <div>{usingNextModelsUI ? 'enabled' : 'disabled'}</div>;
    };

    renderTestComponent(<PreviewComponent />);

    expect(screen.getByText('enabled')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('switch'));

    expect(within(screen.getByRole('dialog')).getByText(/your feedback is invaluable/)).toBeInTheDocument();
  });

  test('it should disable and enable using switch', async () => {
    mockSeenPromoModal();

    const PreviewComponent = () => {
      const { usingNextModelsUI } = useNextModelsUIContext();
      return <div>{usingNextModelsUI ? 'enabled' : 'disabled'}</div>;
    };

    renderTestComponent(<PreviewComponent />);

    expect(screen.getByText('enabled')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('switch'));

    await userEvent.click(screen.getByText('Disable'));

    expect(screen.getByText('disabled')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('switch'));

    expect(screen.getByText('enabled')).toBeInTheDocument();
  });
});

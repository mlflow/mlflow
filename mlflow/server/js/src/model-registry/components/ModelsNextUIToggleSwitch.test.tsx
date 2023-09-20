import userEvent from '@testing-library/user-event';
import { act, renderWithIntl, screen, within } from '../../common/utils/TestUtils';
import { useNextModelsUIContext, withNextModelsUIContext } from '../hooks/useNextModelsUI';
import { ModelsNextUIToggleSwitch } from './ModelsNextUIToggleSwitch';

jest.mock('../../common/utils/FeatureUtils', () => ({
  shouldUseToggleModelsNextUI: () => true,
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
    jest
      .spyOn(window.localStorage, 'getItem')
      .mockImplementation((key) => (key.match(/promo/) ? 'true' : ''));

  const mockUnseenPromoModal = () =>
    jest
      .spyOn(window.localStorage, 'getItem')
      .mockImplementation((key) => (key.match(/promo/) ? 'false' : ''));

  test('it should render the switch and display the promo modal', () => {
    mockUnseenPromoModal();

    renderTestComponent();
    expect(screen.getByRole('switch', { name: /New model registry UI/ })).toBeInTheDocument();
    expect(
      within(screen.getByRole('dialog')).getByText(
        'Introducing new way to manage model deployment',
      ),
    ).toBeInTheDocument();
  });

  test("it should not render display the promo modal when it's already seen", () => {
    mockSeenPromoModal();
    renderTestComponent();
    expect(screen.getByRole('switch')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('it should properly change the flag to true using modal', async () => {
    mockUnseenPromoModal();

    const PreviewComponent = () => {
      const { usingNextModelsUI } = useNextModelsUIContext();
      return <div>{usingNextModelsUI ? 'enabled' : 'disabled'}</div>;
    };

    renderTestComponent(<PreviewComponent />);

    expect(screen.getByText('disabled')).toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByText('Try it now'));
    });

    expect(screen.getByText('enabled')).toBeInTheDocument();
  });

  test('it should properly change the flag to true using switch', async () => {
    mockSeenPromoModal();

    const PreviewComponent = () => {
      const { usingNextModelsUI } = useNextModelsUIContext();
      return <div>{usingNextModelsUI ? 'enabled' : 'disabled'}</div>;
    };

    renderTestComponent(<PreviewComponent />);

    expect(screen.getByText('disabled')).toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('switch'));
    });

    expect(screen.getByText('enabled')).toBeInTheDocument();
  });

  test('it should display confirmation modal when trying to switch off', async () => {
    mockSeenPromoModal();

    const PreviewComponent = () => {
      const { usingNextModelsUI } = useNextModelsUIContext();
      return <div>{usingNextModelsUI ? 'enabled' : 'disabled'}</div>;
    };

    renderTestComponent(<PreviewComponent />);

    expect(screen.getByText('disabled')).toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('switch'));
    });

    expect(screen.getByText('enabled')).toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('switch'));
    });

    expect(
      within(screen.getByRole('dialog')).getByText(/we would love to get your feedback/),
    ).toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByText('Disable'));
    });

    expect(screen.getByText('disabled')).toBeInTheDocument();
  });
});

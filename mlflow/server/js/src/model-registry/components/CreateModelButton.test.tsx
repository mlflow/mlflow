import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { CreateModelButton } from './CreateModelButton';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

const minimalProps = {};

const mockStore = configureStore([thunk, promiseMiddleware()]);
const minimalStore = mockStore({});

const buttonDataTestId = 'create-model-button';
const inputModelDataTestId = 'mlflow-input-modal';

describe('CreateModelButton', () => {
  test('should render with minimal props and store without exploding', () => {
    renderWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CreateModelButton {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(screen.getByTestId(buttonDataTestId)).toBeInTheDocument();
  });

  test('should render button type link correctly', () => {
    const { container } = renderWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CreateModelButton buttonType="link" />
        </MemoryRouter>
      </Provider>,
    );

    expect(container.querySelector('.ant-btn-link')).toBeInTheDocument();
  });

  test('should hide modal by default', () => {
    renderWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CreateModelButton buttonType="link" />
        </MemoryRouter>
      </Provider>,
    );

    expect(screen.queryByTestId(inputModelDataTestId)).not.toBeInTheDocument();
  });

  test('should show modal after button click', async () => {
    renderWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CreateModelButton buttonType="link" />
        </MemoryRouter>
      </Provider>,
    );

    await userEvent.click(screen.getByTestId(buttonDataTestId));
    expect(screen.getByTestId(inputModelDataTestId)).toBeVisible();
  });
});

/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ModelPage } from './ModelPage';

describe('ModelPage', () => {
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const navigate = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
    minimalProps = {
      searchModelVersionsApi: jest.fn().mockResolvedValue({}),
      getRegisteredModelDetailsApi: jest.fn(),
      navigate,
    };
    const versions = [mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY)];
    minimalStore = mockStore({
      entities: {
        modelByName: {
          // @ts-expect-error TS(2345): Argument of type '{ name: any; creation_timestamp:... Remove this comment to see the full error message
          'Model A': mockRegisteredModelDetailed('Model A', versions),
        },
        modelVersionsByModel: {
          'Model A': {
            1: mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
          },
        },
      },
      apis: {},
    });
  });

  const wrapProviders = (node: React.ReactNode) => (
    <MemoryRouter>
      <Provider store={minimalStore}>{node}</Provider>
    </MemoryRouter>
  );

  const renderWithProviders = (node: React.ReactNode) => renderWithIntl(wrapProviders(node));

  test('should render with minimal props and store without exploding', async () => {
    renderWithProviders(<ModelPage {...minimalProps} />);
    // spinner is displayed while fetching data
    expect(screen.getByAltText('Page loading...')).toBeInTheDocument();
  });
});

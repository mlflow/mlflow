/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelVersionTable } from './ModelVersionTable';
import { mockModelVersionDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Table } from 'antd';
import { RegisteringModelDocUrl } from '../../common/constants';
import { act, mountWithIntl, renderWithIntl, screen } from '../../common/utils/TestUtils';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { useNextModelsUIContext, withNextModelsUIContext } from '../hooks/useNextModelsUI';
import { ModelsNextUIToggleSwitch } from './ModelsNextUIToggleSwitch';
import userEvent from '@testing-library/user-event';

jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../common/utils/FeatureUtils'),
  // Force-enable toggling new models UI for test purposes
  shouldUseToggleModelsNextUI: () => true,
}));

describe('ModelVersionTable', () => {
  let wrapper;
  let minimalProps: any;
  beforeEach(() => {
    minimalProps = {
      modelName: 'Model A',
      modelVersions: [],
      onChange: jest.fn(),
    };
  });

  // Wrap the rendered components with redux provider so useDispatch() will work properly
  const mockStoreFactory = configureStore([thunk, promiseMiddleware()]);
  const mountWithProviders = (node: React.ReactNode) =>
    mountWithIntl(
      <MemoryRouter>
        <Provider store={mockStoreFactory({})}>{node}</Provider>
      </MemoryRouter>,
    );

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithProviders(<ModelVersionTable {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
  test('should render correct empty text', () => {
    wrapper = wrapper = mountWithProviders(<ModelVersionTable {...minimalProps} />);
    expect(wrapper.find(`a[href="${RegisteringModelDocUrl}"]`)).toHaveLength(1);
  });
  test('should render active versions when activeStageOnly is true', () => {
    const props = {
      ...minimalProps,
      modelVersions: [
        mockModelVersionDetailed('Model A', 1, Stages.NONE, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 2, Stages.PRODUCTION, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 3, Stages.STAGING, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 4, Stages.ARCHIVED, ModelVersionStatus.READY),
      ],
    };
    wrapper = mountWithProviders(<ModelVersionTable {...props} />);
    expect(wrapper.find(Table).props().dataSource.length).toBe(4);
    const propsWithActive = {
      ...props,
      activeStageOnly: true,
    };
    wrapper = mountWithProviders(<ModelVersionTable {...propsWithActive} />);
    expect(wrapper.find(Table).props().dataSource.length).toBe(2);
  });

  test('should display aliases column instead of stage when new models UI is used', async () => {
    const TestComponent = withNextModelsUIContext(() => {
      const { usingNextModelsUI } = useNextModelsUIContext();
      return (
        <Provider store={mockStoreFactory({})}>
          {/* <ModelVersionTable> relies on usingNextModelsUI prop instead of getting it from context by itself */}
          <ModelVersionTable {...minimalProps} usingNextModelsUI={usingNextModelsUI} />
          <ModelsNextUIToggleSwitch />
        </Provider>
      );
    });
    renderWithIntl(<TestComponent />);

    // Assert stage column being visible and aliases column being absent
    expect(screen.queryByRole('columnheader', { name: 'Stage' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Aliases' })).not.toBeInTheDocument();

    // Flip the "Next models UI" switch
    await act(async () => {
      userEvent.click(screen.getByRole('switch'));
    });

    // Assert the opposite: stage column should be invisible and aliases column should be present
    expect(screen.queryByRole('columnheader', { name: 'Stage' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Aliases' })).toBeInTheDocument();
  });
});

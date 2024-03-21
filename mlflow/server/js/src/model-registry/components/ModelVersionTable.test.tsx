import React from 'react';
import { mockModelVersionDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { renderWithIntl, screen } from 'common/utils/TestUtils.react17';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { ModelVersionTable } from './ModelVersionTable';

jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../common/utils/FeatureUtils'),
  // Force-enable toggling new models UI for test purposes
  shouldShowModelsNextUI: () => true,
}));

describe('ModelVersionTable', () => {
  const minimalProps = {
    modelName: 'Model A',
    modelVersions: [],
    onChange: jest.fn(),
    onMetadataUpdated: jest.fn(),
    usingNextModelsUI: false,
    aliases: [],
  };

  const mockStoreFactory = configureStore([thunk, promiseMiddleware()]);

  // Wrap the rendered components with redux provider so useDispatch() will work properly
  const wrapProviders = (node: React.ReactNode) => (
    <MemoryRouter>
      <Provider store={mockStoreFactory({})}>{node}</Provider>
    </MemoryRouter>
  );

  const renderWithProviders = (node: React.ReactNode) => renderWithIntl(wrapProviders(node));

  test('should render with minimal props', () => {
    renderWithProviders(<ModelVersionTable {...minimalProps} />);
    expect(screen.getByRole('row')).toBeInTheDocument();
  });
  test('should render correct empty text', () => {
    renderWithProviders(<ModelVersionTable {...minimalProps} />);
    expect(screen.getByRole('link', { name: /Learn more/ })).toBeInTheDocument();
  });
  test('should render active versions when activeStageOnly is true', () => {
    const props = {
      ...minimalProps,
      modelVersions: [
        mockModelVersionDetailed('Model A', 1, Stages.NONE, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 2, Stages.PRODUCTION, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 3, Stages.STAGING, ModelVersionStatus.READY),
        mockModelVersionDetailed('Model A', 4, Stages.ARCHIVED, ModelVersionStatus.READY),
      ] as any,
    };
    const { rerender } = renderWithProviders(<ModelVersionTable {...props} />);

    expect(screen.getAllByRole('row')).toHaveLength(5); // 4 model rows and 1 header row

    rerender(wrapProviders(<ModelVersionTable {...props} activeStageOnly />));

    expect(screen.getAllByRole('row')).toHaveLength(3); // 2 model rows and 1 header row
  });
  test('should display stages instead of aliases and tags when new models UI is not used', () => {
    renderWithProviders(<ModelVersionTable {...minimalProps} />);
    expect(screen.queryByRole('columnheader', { name: 'Stage' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Aliases' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Tags' })).not.toBeInTheDocument();
  });

  test('should display aliases and tags column instead of stage when new models UI is used', () => {
    const modelName = 'Random Forest Model';
    const props = {
      ...minimalProps,
      modelName: modelName,
      usingNextModelsUI: true,
      modelVersions: [mockModelVersionDetailed(modelName, 1, Stages.NONE, ModelVersionStatus.READY)],
      aliases: [{alias: 'champion', version: '1'}]
    };
    renderWithProviders(<ModelVersionTable {...props} />);
    expect(screen.queryByRole('columnheader', { name: 'Stage' })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Aliases' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Tags' })).toBeInTheDocument();
    expect(screen.queryByText(/champion/)).toBeInTheDocument();
  });
});

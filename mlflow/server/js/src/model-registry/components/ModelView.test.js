import React from 'react';
import { mount, shallow } from 'enzyme';
import { ModelView, StageFilters } from './ModelView';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { BrowserRouter } from 'react-router-dom';
import { ModelVersionTable } from './ModelVersionTable';
import Utils from '../../common/utils/Utils';
import { getCompareModelVersionsPageRoute } from '../routes';

describe('ModelView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let historyMock;
  const mockModel = {
    name: 'Model A',
    latestVersions: [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
      mockModelVersionDetailed('Model A', 2, Stages.STAGING, ModelVersionStatus.READY),
      mockModelVersionDetailed('Model A', 3, Stages.NONE, ModelVersionStatus.READY),
    ],
    versions: [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY, []),
      mockModelVersionDetailed('Model A', 2, Stages.STAGING, ModelVersionStatus.READY, []),
      mockModelVersionDetailed('Model A', 3, Stages.NONE, ModelVersionStatus.READY, []),
    ],
  };

  beforeEach(() => {
    historyMock = jest.fn();
    minimalProps = {
      model: mockRegisteredModelDetailed(
        mockModel.name,
        mockModel.latestVerions,
        mockModel.permissionLevel,
      ),
      modelVersions: mockModel.versions,
      handleEditDescription: jest.fn(),
      handleDelete: jest.fn(),
      showEditPermissionModal: jest.fn(),
      history: { push: historyMock },
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mount(
      <BrowserRouter>
        <ModelView {...minimalProps} />
      </BrowserRouter>,
    );
    expect(wrapper.find(ModelView).length).toBe(1);
  });

  test('should render all model versions initially', () => {
    wrapper = mount(
      <BrowserRouter>
        <ModelView {...minimalProps} />
      </BrowserRouter>,
    );
    expect(wrapper.find('td.model-version').length).toBe(3);
    expect(
      wrapper
        .find('td.model-version')
        .at(0)
        .text(),
    ).toBe('Version 1');
    expect(
      wrapper
        .find('td.model-version')
        .at(1)
        .text(),
    ).toBe('Version 2');
    expect(
      wrapper
        .find('td.model-version')
        .at(2)
        .text(),
    ).toBe('Version 3');
  });

  test('should render model version table with activeStageOnly when "Active" button is on', () => {
    wrapper = mount(
      <BrowserRouter>
        <ModelView {...minimalProps} />
      </BrowserRouter>,
    );
    expect(wrapper.find(ModelVersionTable).props().activeStageOnly).toBe(false);
    instance = wrapper.find(ModelView).instance();
    instance.setState({ stageFilter: StageFilters.ACTIVE });
    wrapper.update();
    expect(wrapper.find(ModelVersionTable).props().activeStageOnly).toBe(true);
  });

  test('Page title is set', () => {
    const mockUpdatePageTitle = jest.fn();
    Utils.updatePageTitle = mockUpdatePageTitle;
    wrapper = shallow(<ModelView {...minimalProps} />);
    expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Model A - MLflow Model');
  });

  test('should disable dropdown delete menu item when model has active versions', () => {
    const props = {
      ...minimalProps,
      model: {
        ...minimalProps.model,
      },
    };
    wrapper = mount(
      <BrowserRouter>
        <ModelView {...props} />
      </BrowserRouter>,
    );
    wrapper
      .find('.breadcrumb-dropdown')
      .hostNodes()
      .simulate('click');
    // The antd `Menu.Item` component converts the `disabled` attribute to `aria-disabled`
    // when generating HTML. Accordingly, we check for the presence of the `aria-disabled`
    // attribute within the rendered HTML.
    const deleteMenuItem = wrapper.find('.delete').hostNodes();
    expect(deleteMenuItem.prop('aria-disabled')).toBe(true);
    deleteMenuItem.simulate('click');
    expect(wrapper.find(ModelView).instance().state.isDeleteModalVisible).toBe(false);
  });

  test('compare button is disabled when no/1 run selected, active when 2+ runs selected', () => {
    wrapper = mount(
      <BrowserRouter>
        <ModelView {...minimalProps} />
      </BrowserRouter>,
    );

    expect(wrapper.find('.btn').length).toBe(1);
    expect(wrapper.find('.btn').props().disabled).toEqual(true);

    wrapper
      .find(ModelView)
      .instance()
      .setState({
        runsSelected: { run_id_1: 'version_1' },
      });
    wrapper.update();
    expect(wrapper.find('.btn').props().disabled).toEqual(true);

    const twoRunsSelected = { run_id_1: 'version_1', run_id_2: 'version_2' };
    wrapper
      .find(ModelView)
      .instance()
      .setState({
        runsSelected: twoRunsSelected,
      });
    wrapper.update();
    expect(wrapper.find('.btn').props().disabled).toEqual(false);

    wrapper.find('.btn').simulate('click');
    expect(historyMock).toHaveBeenCalledWith(
      getCompareModelVersionsPageRoute(minimalProps['model']['name'], twoRunsSelected),
    );
  });
});

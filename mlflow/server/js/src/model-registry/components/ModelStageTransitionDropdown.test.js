import React from 'react';
import { shallow } from 'enzyme';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Dropdown } from 'antd';
import { mockGetFieldValue, Stages, stageTagComponents, modelStageNames } from '../test-utils';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('ModelStageTransitionDropdown', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      currentStage: Stages.NONE,
      stageTagComponents: stageTagComponents(),
      modelStageNames: modelStageNames
    };
    commonProps = {
      ...minimalProps,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(<ModelStageTransitionDropdown {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should omit current stage in dropdown', () => {
    const props = {
      ...commonProps,
      currentStage: Stages.STAGING,
    };
    wrapper = mountWithIntl(<ModelStageTransitionDropdown {...props} />);
    wrapper
      .find('.stage-transition-dropdown')
      .first()
      .simulate('click');
    const menuHtml = mountWithIntl(wrapper.find(Dropdown).props().overlay).html();

    expect(menuHtml).not.toContain(Stages.STAGING);
    expect(menuHtml).toContain(Stages.PRODUCTION);
    expect(menuHtml).toContain(Stages.NONE);
    expect(menuHtml).toContain(Stages.ARCHIVED);
  });

  test('handleMenuItemClick - archiveExistingVersions', () => {
    const mockOnSelect = jest.fn();
    const props = {
      ...commonProps,
      onSelect: mockOnSelect,
    };
    const activity = {};
    wrapper = shallow(<ModelStageTransitionDropdown {...props} />);
    const mockArchiveFieldValues = [true, false, undefined];
    mockArchiveFieldValues.forEach((fieldValue) => {
      const expectArchiveFieldValue = Boolean(fieldValue); // undefined should become false also
      const instance = wrapper.instance();
      instance.transitionFormRef = {
        current: {
          getFieldValue: mockGetFieldValue('', fieldValue),
          resetFields: () => {},
        },
      };
      instance.handleMenuItemClick(activity);
      instance.state.handleConfirm();
      expect(mockOnSelect).toHaveBeenCalledWith(activity, expectArchiveFieldValue);
    });
  });
});

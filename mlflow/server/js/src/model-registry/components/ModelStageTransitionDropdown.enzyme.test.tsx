/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Stages } from '../constants';
import { Dropdown } from '@databricks/design-system';
import { mockGetFieldValue } from '../test-utils';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('ModelStageTransitionDropdown', () => {
  let wrapper: any;
  let minimalProps: any;
  let commonProps: any;

  beforeEach(() => {
    minimalProps = {
      currentStage: Stages.NONE,
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
    wrapper.find('.mlflow-stage-transition-dropdown').first().simulate('click');
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
      instance.state.handleConfirm({
        archiveExistingVersions: fieldValue,
      });
      // eslint-disable-next-line jest/no-standalone-expect
      expect(mockOnSelect).toHaveBeenCalledWith(activity, expectArchiveFieldValue);
    });
  });
});

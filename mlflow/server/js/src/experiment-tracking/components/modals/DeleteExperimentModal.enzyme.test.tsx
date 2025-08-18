/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { DeleteExperimentModalImpl } from './DeleteExperimentModal';
import { ConfirmModal } from './ConfirmModal';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';

describe('DeleteExperimentModal', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;
  const fakeExperimentId = 'fakeExpId';

  const navigate = jest.fn();

  beforeEach(() => {
    navigate.mockClear();
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      activeExperimentIds: ['0'],
      experimentId: '0',
      experimentName: 'myFirstExperiment',
      deleteExperimentApi: (experimentId: any, deleteExperimentRequestId: any) => {
        const response = { value: { experiment_id: fakeExperimentId } };
        return Promise.resolve(response);
      },
      navigate,
    };
    wrapper = shallow(<DeleteExperimentModalImpl {...minimalProps} />);
  });
  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<DeleteExperimentModalImpl {...minimalProps} />);
    expect(wrapper.find(ConfirmModal).length).toBe(1);
    expect(wrapper.length).toBe(1);
  });
  test('handleSubmit redirects user to root page if current experiment is the only active experiment', async () => {
    instance = wrapper.instance();
    await instance.handleSubmit();
    expect(navigate).toHaveBeenCalledWith(createMLflowRoutePath('/'));
  });
  test('handleSubmit redirects to compare experiment page if current experiment is one of several active experiments', async () => {
    const props = Object.assign({}, minimalProps, { activeExperimentIds: ['0', '1', '2'] });
    instance = shallow(<DeleteExperimentModalImpl {...props} />).instance();
    await instance.handleSubmit();

    expect(navigate).toHaveBeenCalledWith(createMLflowRoutePath('/compare-experiments/s?experiments=["1","2"]'));
  });
  test('handleSubmit does not perform redirection if DeleteExperiment request fails', async () => {
    const props = {
      ...minimalProps,
      deleteExperimentApi: () => Promise.reject(new Error('DeleteExperiment failed!')),
    };
    wrapper = shallow(<DeleteExperimentModalImpl {...props} />);
    instance = wrapper.instance();
    await instance.handleSubmit();

    expect(navigate).not.toHaveBeenCalled();
  });
  test('handleSubmit does not perform redirection if deleted experiment is not active experiment', async () => {
    wrapper = shallow(<DeleteExperimentModalImpl {...{ ...minimalProps, activeExperimentIds: undefined }} />);
    instance = wrapper.instance();
    await instance.handleSubmit();

    expect(navigate).not.toHaveBeenCalled();
  });
});

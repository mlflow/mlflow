/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { CreateExperimentModalImpl } from './CreateExperimentModal';
import { GenericInputModal } from './GenericInputModal';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';

describe('CreateExperimentModal', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;

  const navigate = jest.fn();

  const fakeExperimentId = 'fakeExpId';
  beforeEach(() => {
    navigate.mockClear();
    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      experimentNames: [],
      createExperimentApi: (experimentName: any, artifactLocation: any) => {
        const response = { value: { experiment_id: fakeExperimentId } };
        return Promise.resolve(response);
      },
      searchExperimentsApi: () => Promise.resolve([]),
      onExperimentCreated: jest.fn(),
      navigate,
    };
    wrapper = shallow(<CreateExperimentModalImpl {...minimalProps} />);
  });
  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<CreateExperimentModalImpl {...minimalProps} />);
    expect(wrapper.find(GenericInputModal).length).toBe(1);
    expect(wrapper.length).toBe(1);
  });
  test('handleCreateExperiment redirects user to newly-created experiment page', async () => {
    instance = wrapper.instance();
    await instance.handleCreateExperiment({
      experimentName: 'myNewExp',
      artifactLocation: 'artifactLoc',
    });

    expect(navigate).toHaveBeenCalledWith(createMLflowRoutePath('/experiments/fakeExpId'));
  });
  test('handleCreateExperiment does not perform redirection if API requests fail', async () => {
    const propsVals = [
      {
        ...minimalProps,
        createExperimentApi: () => Promise.reject(new Error('CreateExperiment failed!')),
      },
    ];
    const testPromises: any = [];
    propsVals.forEach(async (props) => {
      wrapper = shallow(<CreateExperimentModalImpl {...props} />);
      instance = wrapper.instance();
      const payload = { experimentName: 'myNewExp', artifactLocation: 'artifactLoc' };
      testPromises.push(await expect(instance.handleCreateExperiment(payload)).rejects.toThrow());
    });
    await Promise.all(testPromises);

    expect(navigate).not.toHaveBeenCalled();
  });
});

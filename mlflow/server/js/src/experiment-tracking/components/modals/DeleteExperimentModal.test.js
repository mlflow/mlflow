import React from 'react';
import { shallow } from 'enzyme';
import { DeleteExperimentModalImpl } from './DeleteExperimentModal';
import { ConfirmModal } from './ConfirmModal';

describe('DeleteExperimentModal', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let location = {};
  const fakeExperimentId = 'fakeExpId';

  beforeEach(() => {
    location = { search: 'initialSearchValue' };
    const history = {
      push: (url) => {
        location.search = url;
      },
    };

    minimalProps = {
      isOpen: false,
      onClose: jest.fn(),
      activeExperimentId: '0',
      experimentId: '0',
      experimentName: 'myFirstExperiment',
      deleteExperimentApi: (experimentId, deleteExperimentRequestId) => {
        const response = { value: { experiment_id: fakeExperimentId } };
        return Promise.resolve(response);
      },
      listExperimentsApi: () => Promise.resolve([]),
      history: history,
    };
    wrapper = shallow(<DeleteExperimentModalImpl {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<DeleteExperimentModalImpl {...minimalProps} />);
    expect(wrapper.find(ConfirmModal).length).toBe(1);
    expect(wrapper.length).toBe(1);
  });

  test('handleSubmit redirects user to root page if active experiment is current experiment', (done) => {
    instance = wrapper.instance();
    instance.handleSubmit().then(() => {
      expect(location.search).toEqual('/');
      done();
    });
  });

  test('handleSubmit does not perform redirection if DeleteExperiment request fails', (done) => {
    const props = {
      ...minimalProps,
      deleteExperimentApi: () => Promise.reject(new Error('DeleteExperiment failed!')),
    };
    wrapper = shallow(<DeleteExperimentModalImpl {...props} />);
    instance = wrapper.instance();
    instance.handleSubmit().then(() => {
      expect(location.search).toEqual('initialSearchValue');
      done();
    });
  });

  test('handleSubmit does not perform redirection if deleted experiment is not active experiment', (done) => {
    wrapper = shallow(
      <DeleteExperimentModalImpl {...{ ...minimalProps, activeExperimentId: undefined }} />,
    );
    instance = wrapper.instance();
    instance.handleSubmit().then(() => {
      expect(location.search).toEqual('initialSearchValue');
      done();
    });
  });
});

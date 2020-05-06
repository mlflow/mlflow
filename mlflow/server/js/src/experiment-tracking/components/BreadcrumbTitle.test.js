import React from 'react';
import { shallow } from 'enzyme';
import { BreadcrumbTitle } from './BreadcrumbTitle';
import { Experiment } from '../sdk/MlflowMessages';

describe('BreadcrumbTitle', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  const mockExperimentId = 1234;
  const mockExperimentName = 'experimentName';
  const runUuids = ['RunID1', 'RunID2'];
  const runNames = ['RunName1', 'RunName2'];

  const mockExperiment = Experiment.fromJs({
    experiment_id: mockExperimentId,
    name: mockExperimentName,
  });
  const mockTitle = 'mockTitle';

  beforeEach(() => {
    minimalProps = { experiment: mockExperiment, title: mockTitle };
    commonProps = { ...minimalProps };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<BreadcrumbTitle {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render correct experiment name', () => {
    wrapper = shallow(<BreadcrumbTitle {...minimalProps} />);
    expect(wrapper.childAt(0).prop('children')).toBe(mockExperimentName);
  });

  test('should render correct experiment title', () => {
    wrapper = shallow(<BreadcrumbTitle {...minimalProps} />);
    expect(wrapper.find('span').text()).toBe(mockTitle);
  });

  test('should render a single runName when a single runUuid is passed', () => {
    commonProps['runUuids'] = [runUuids[0]];
    commonProps['runNames'] = [runNames[0]];
    wrapper = shallow(<BreadcrumbTitle {...commonProps} />);
    expect(wrapper.childAt(2).prop('children')).toBe(runNames[0]);
  });

  test('should render comparing runs when multiple runUuids are passed', () => {
    commonProps['runUuids'] = runUuids;
    commonProps['runNames'] = runNames;
    const expectedString = 'Comparing ' + runUuids.length + ' Runs';
    wrapper = shallow(<BreadcrumbTitle {...commonProps} />);
    expect(
      wrapper
        .childAt(2)
        .prop('children')
        .join(''),
    ).toBe(expectedString);
  });
});

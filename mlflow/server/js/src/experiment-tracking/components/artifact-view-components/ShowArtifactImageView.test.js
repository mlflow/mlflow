import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactImageView from './ShowArtifactImageView';
import { LazyPlot } from '../LazyPlot';

describe('ShowArtifactImageView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakepath',
      runUuid: 'fakeUuid',
    };
    commonProps = { ...minimalProps };
    wrapper = shallow(<ShowArtifactImageView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactImageView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-image-view-loading').length).toBe(1);
  });

  test('should render gif image in container', () => {
    wrapper.setState({ loading: false });
    wrapper.setProps({ path: 'fake.gif', runUuid: 'fakeRunId' });
    expect(wrapper.find('.image-outer-container')).toHaveLength(1);
    expect(wrapper.find('.image-container')).toHaveLength(1);
    expect(wrapper.find('img')).toHaveLength(1);
  });

  test('should render static image in plotly component', () => {
    wrapper = shallow(<ShowArtifactImageView path='fake.png' runUuid='fakerunuuid' />);
    wrapper.setState({ loading: false });
    expect(wrapper.find(LazyPlot)).toHaveLength(1);
  });

  test('should call fetchImage on component update', () => {
    instance = wrapper.instance();
    instance.fetchImage = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchImage).toBeCalled();
  });
});

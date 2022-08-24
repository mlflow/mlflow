import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactTableView from './ShowArtifactTableView';

describe('ShowArtifactImageView', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakePath',
      runUuid: 'fakeUuid',
    };
    commonProps = { ...minimalProps };
    wrapper = shallow(<ShowArtifactTableView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactTableView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

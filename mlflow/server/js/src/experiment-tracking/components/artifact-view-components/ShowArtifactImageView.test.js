import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactImageView from './ShowArtifactImageView';

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
    wrapper = shallow(<ShowArtifactImageView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactImageView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});

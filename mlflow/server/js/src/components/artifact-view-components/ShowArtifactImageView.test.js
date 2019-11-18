import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactImageView from './ShowArtifactImageView';

describe('Render test', () => {
  it('renders empty image container', () => {
    const wrapper = shallow(<ShowArtifactImageView path="fakepath" runUuid="fakerunuuid" />);

    wrapper.setState({ loading: false });

    expect(wrapper.find('.image-outer-container')).toHaveLength(1);
    expect(wrapper.find('.image-container')).toHaveLength(1);
  });

  it('renders gif image', () => {
    const wrapper = shallow(<ShowArtifactImageView path="fake.gif" runUuid="fakerunuuid" />);

    wrapper.setState({ loading: false });

    expect(wrapper.find('img')).toHaveLength(1);
  });

  it('renders static image', () => {
    const wrapper = shallow(<ShowArtifactImageView path="fake.png" runUuid="fakerunuuid" />);

    wrapper.setState({ loading: false });

    expect(wrapper.find('PlotlyComponent')).toHaveLength(1);
  });
});

require('css.escape');
import React from 'react';
import { mount } from 'enzyme';
import { getUUID } from '../../Actions';
import ShowArtifactImageView from './ShowArtifactImageView';

describe('<ShowArtifactImageView />', () => {
  let wrapper;
  let uuid = getUUID();

  test('should render with image url ', () => {
    const props = {runUuid: uuid, path: "test.jpg"}
    wrapper = mount(<ShowArtifactImageView {...props} />);
    expect(wrapper.find('.image-container').html().includes("style=\"background-image:")).toBeTruthy();
  });

  test('should render with image url when CSS special charactor', () => {
    const props = {runUuid: uuid, path: "test( .jpg"}
    wrapper = mount(<ShowArtifactImageView {...props} />);
    expect(wrapper.find('.image-container').html().includes("style=\"background-image:")).toBeTruthy();
  });
});

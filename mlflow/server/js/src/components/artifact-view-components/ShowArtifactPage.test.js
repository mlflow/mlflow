require('css.escape');
import React from 'react';
import { shallow } from 'enzyme';
import { getUUID } from '../../Actions';
import ShowArtifactPage, { getSrc } from './ShowArtifactPage';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';

describe('#getSrc', () => {
  let ret;
  let expected;
  let uuid = getUUID();

  test('should return URI escaped value when no third argument given', () => {
    ret = getSrc("test .jpg", uuid)
    expected = "get-artifact?path=test%20.jpg&run_uuid=" + uuid;
    expect(ret).toEqual(expected);
  });

  test('should return URI escaped value when third argument is false', () => {
    ret = getSrc("test .jpg", uuid, false)
    expected = "get-artifact?path=test%20.jpg&run_uuid=" + uuid;
    expect(ret).toEqual(expected);
  });

  test('should return CSS escaped value when third argument is true', () => {
    ret = getSrc("test .jpg", uuid, true)
    expected = "get-artifact?path=test\\ \\.jpg&run_uuid=" + uuid;
    expect(ret).toEqual(expected);
  });
});

describe('<ShowArtifactPage />', () => {
  let wrapper;
  let uuid = getUUID();

  test('should render ShowArtifactImageView when path is image path', () => {
    const props = {runUuid: uuid, path: "huga.jpg"}
    wrapper = shallow(<ShowArtifactPage {...props} />);
    expect(wrapper.find(ShowArtifactImageView).length).toBe(1);
  });

  test('should render ShowArtifactImageView when path is image path includes special character', () => {
    const props = {runUuid: uuid, path: "huga .jpg"}
    wrapper = shallow(<ShowArtifactPage {...props} />);
    expect(wrapper.find(ShowArtifactImageView).length).toBe(1);
  });

  test('should render ShowArtifactTextView when path is text path', () => {
    const props = {runUuid: uuid, path: "huga.txt"}
    wrapper = shallow(<ShowArtifactPage {...props} />);
    expect(wrapper.find(ShowArtifactTextView).length).toBe(1);
  });

  test('should render ShowArtifactHtmlView when path is html path', () => {
    const props = {runUuid: uuid, path: "huga.html"}
    wrapper = shallow(<ShowArtifactPage {...props} />);
    expect(wrapper.find(ShowArtifactHtmlView).length).toBe(1);
  });
});

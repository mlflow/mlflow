import React from 'react';
import { mount, shallow } from 'enzyme';
import ShowArtifactImageView from './ShowArtifactImageView';

describe('ShowArtifactImageView', () => {
  let wrapper;
  let minimalProps;
  let objectUrlSpy;

  beforeAll(() => {
    objectUrlSpy = jest
      .spyOn(window.URL, 'createObjectURL')
      .mockImplementation(() => 'blob:abc-12345');
  });

  afterAll(() => {
    objectUrlSpy.mockRestore();
  });

  beforeEach(() => {
    minimalProps = {
      path: 'fakePath',
      runUuid: 'fakeUuid',
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactImageView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should fetch image as an XHR', (done) => {
    const getArtifact = jest.fn().mockImplementation(() => Promise.resolve(new ArrayBuffer(8)));
    wrapper = mount(<ShowArtifactImageView {...minimalProps} getArtifact={getArtifact} />);
    expect(getArtifact).toBeCalledWith('get-artifact?path=fakePath&run_uuid=fakeUuid');

    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('img[src="blob:abc-12345"]').length).toBeTruthy();
      done();
    });
  });
});

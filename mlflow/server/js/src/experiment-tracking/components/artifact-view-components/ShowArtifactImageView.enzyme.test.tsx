/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { describe, beforeAll, jest, afterAll, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import { mount, shallow } from 'enzyme';
import ShowArtifactImageView from './ShowArtifactImageView';

describe('ShowArtifactImageView', () => {
  let wrapper: any;
  let minimalProps: any;
  let objectUrlSpy: any;

  beforeAll(() => {
    objectUrlSpy = jest.spyOn(window.URL, 'createObjectURL').mockImplementation(() => 'blob:abc-12345');
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

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should fetch image as an XHR', (done) => {
    const getArtifact = jest.fn(() => Promise.resolve(new ArrayBuffer(8)));
    wrapper = mount(<ShowArtifactImageView {...minimalProps} getArtifact={getArtifact} />);
    // @ts-expect-error Expected 0 arguments, but got 1
    expect(getArtifact).toHaveBeenCalledWith(expect.stringMatching(/get-artifact\?path=fakePath&run_uuid=fakeUuid/));

    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('img[src="blob:abc-12345"]').length).toBeTruthy();
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should constrain image preview to its parent width', (done) => {
    const getArtifact = jest.fn(() => Promise.resolve(new ArrayBuffer(8)));
    wrapper = mount(<ShowArtifactImageView {...minimalProps} getArtifact={getArtifact} />);

    setImmediate(() => {
      try {
        wrapper.update();
        const image = wrapper.find('img[src="blob:abc-12345"]').at(0);
        const imageWrapper = image.parent();

        expect(imageWrapper.prop('css')).toEqual(
          expect.objectContaining({
            display: 'block',
            width: '100%',
          }),
        );
        expect(image.prop('css')).toEqual(
          expect.objectContaining({
            display: 'block',
            maxWidth: '100%',
          }),
        );
        done();
      } catch (error) {
        done(error);
      }
    });
  });
});

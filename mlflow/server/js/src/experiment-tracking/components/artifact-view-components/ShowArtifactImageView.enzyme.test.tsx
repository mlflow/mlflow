/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { describe, beforeAll, jest, afterAll, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactImageView from './ShowArtifactImageView';
import { ArtifactPreviewTooLargeError } from '../../../common/utils/ArtifactUtils';
import { mountWithIntl } from '../../../common/utils/TestUtils.enzyme';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';

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
      artifactRootUri: 'mlflow-artifacts:/1/fakeUuid/artifacts',
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactImageView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should fetch image as an XHR', async () => {
    const getArtifact = jest.fn(() => Promise.resolve(new ArrayBuffer(8)));
    wrapper = mountWithIntl(<ShowArtifactImageView {...minimalProps} getArtifact={getArtifact} />);
    await new Promise(setImmediate);
    // @ts-expect-error Expected 0 arguments, but got 1
    expect(getArtifact).toHaveBeenCalledWith('ajax-api/2.0/mlflow-artifacts/artifacts/1/fakeUuid/artifacts/fakePath');
    wrapper.update();
    expect(wrapper.find('img[src="blob:abc-12345"]').length).toBeTruthy();
  });

  test('should ignore a stale response after the path changes', async () => {
    let resolveFirst: (value: ArrayBuffer) => void = () => {};
    const firstBytes = new ArrayBuffer(1);
    const secondBytes = new ArrayBuffer(2);
    const getArtifact = jest.fn((url: string) =>
      url.includes('first.svg')
        ? new Promise<ArrayBuffer>((resolve) => {
            resolveFirst = resolve;
          })
        : Promise.resolve(secondBytes),
    );
    objectUrlSpy.mockImplementation((blob: Blob) => (blob.size === 1 ? 'blob:first' : 'blob:second'));

    wrapper = mountWithIntl(<ShowArtifactImageView {...minimalProps} path="first.svg" getArtifact={getArtifact} />);
    wrapper.setProps({ path: 'second.svg' });
    await new Promise((resolve) => setImmediate(resolve));
    wrapper.update();
    expect(wrapper.find('img').first().prop('src')).toBe('blob:second');

    resolveFirst(firstBytes);
    await new Promise((resolve) => setImmediate(resolve));
    wrapper.update();
    expect(wrapper.find('img').first().prop('src')).toBe('blob:second');
    expect(wrapper.find('img').first().prop('alt')).toBe('second.svg');
  });

  test('shows a loading placeholder and a local error for an oversized image', async () => {
    const getArtifact = jest.fn<() => Promise<ArrayBuffer>>().mockRejectedValue(new ArtifactPreviewTooLargeError());
    wrapper = mountWithIntl(<ShowArtifactImageView {...minimalProps} getArtifact={getArtifact} />);
    expect(wrapper.find(ArtifactViewSkeleton)).toHaveLength(1);
    await new Promise(setImmediate);
    wrapper.update();
    expect(getArtifact).toHaveBeenCalledTimes(1);
    expect(wrapper.find(ArtifactViewSkeleton)).toHaveLength(0);
    expect(wrapper.find(ArtifactViewErrorState)).toHaveLength(1);
    expect(wrapper.find('img[src="blob:abc-12345"]')).toHaveLength(0);
  });
});

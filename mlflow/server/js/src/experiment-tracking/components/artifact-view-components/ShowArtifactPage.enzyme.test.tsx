/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import ShowArtifactPage from './ShowArtifactPage';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import { LazyShowArtifactMapView } from './LazyShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import { LazyShowArtifactTableView } from './LazyShowArtifactTableView';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';
import {
  IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS,
  MAP_EXTENSIONS,
  HTML_EXTENSIONS,
  DATA_EXTENSIONS,
  AUDIO_EXTENSIONS,
} from '../../../common/utils/FileUtils';
import { RunTag } from '../../sdk/MlflowMessages';
import { LazyShowArtifactAudioView } from './LazyShowArtifactAudioView';

// Mock these methods because js-dom doesn't implement window.Request
jest.mock('../../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/ArtifactUtils')>('../../../common/utils/ArtifactUtils'),
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
  getArtifactContent: jest.fn().mockResolvedValue(),
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
  getArtifactBytesContent: jest.fn().mockResolvedValue(),
}));

describe('ShowArtifactPage', () => {
  let wrapper: any;
  let minimalProps: any;
  let commonProps;
  beforeEach(() => {
    minimalProps = {
      runUuid: 'fakeUuid',
      artifactRootUri: 'path/to/root/artifact',
    };
    (ShowArtifactPage.prototype as any).fetchArtifacts = jest.fn();
    commonProps = { ...minimalProps, path: 'fakepath' };
    wrapper = mountWithIntl(<ShowArtifactPage {...commonProps} />);
  });
  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(<ShowArtifactPage {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
  test('should render "select to preview" view when path is unspecified', () => {
    wrapper = mountWithIntl(<ShowArtifactPage {...minimalProps} />);
    expect(wrapper.text().includes('Select a file to preview')).toBe(true);
  });
  test('should render "too large to preview" view when size is too large', () => {
    wrapper.setProps({ path: 'file_without_extension', runUuid: 'runId', size: 100000000 });
    expect(wrapper.text().includes('Select a file to preview')).toBe(false);
    expect(wrapper.text().includes('File is too large to preview')).toBe(true);
  });
  test('should render logged model view when path is in runs tag logged model history', () => {
    wrapper.setProps({
      path: 'somePath',
      isDirectory: true,
      runTags: {
        'mlflow.log-model.history': (RunTag as any).fromJs({
          key: 'mlflow.log-model.history',
          value: JSON.stringify([
            {
              run_id: 'run-uuid',
              artifact_path: 'somePath',
              flavors: { keras: {}, python_function: {} },
            },
          ]),
        }),
      },
    });
    expect(wrapper.find(ShowArtifactLoggedModelView).length).toBe(1);
  });
  test('should render logged model view when path is nested in subdirectory', () => {
    wrapper.setProps({
      path: 'dir/somePath',
      isDirectory: true,
      runTags: {
        'mlflow.log-model.history': (RunTag as any).fromJs({
          key: 'mlflow.log-model.history',
          value: JSON.stringify([
            {
              run_id: 'run-uuid',
              artifact_path: 'dir/somePath',
              flavors: { keras: {}, python_function: {} },
            },
          ]),
        }),
      },
    });
    expect(wrapper.find(ShowArtifactLoggedModelView).length).toBe(1);
  });
  test('should render "select to preview" view when path has no extension', () => {
    wrapper.setProps({ path: 'file_without_extension', runUuid: 'runId' });
    expect(wrapper.text().includes('Select a file to preview')).toBe(true);
  });
  test('should render "select to preview" view when path has unknown extension', () => {
    wrapper.setProps({ path: 'file.unknown', runUuid: 'runId' });
    expect(wrapper.text().includes('Select a file to preview')).toBe(true);
  });
  test('should render image view for common image extensions', () => {
    IMAGE_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(ShowArtifactImageView).length).toBe(1);
    });
  });
  test('should render "select to preview" view for folder with common image extensions', () => {
    IMAGE_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId', isDirectory: 'true' });
      expect(wrapper.text().includes('Select a file to preview')).toBe(true);
    });
  });
  test('should render html view for common html extensions', () => {
    HTML_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(ShowArtifactHtmlView).length).toBe(1);
    });
  });
  test('should render map view for common map extensions', () => {
    MAP_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(LazyShowArtifactMapView).length).toBe(1);
    });
  });
  test('should render text view for common text extensions', () => {
    TEXT_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(ShowArtifactTextView).length).toBe(1);
    });
  });
  test('should render data table view for common tabular data extensions', () => {
    DATA_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(LazyShowArtifactTableView).length).toBe(1);
    });
  });
  test('should render audio view for common audio data extensions', () => {
    AUDIO_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(LazyShowArtifactAudioView).length).toBe(1);
    });
  });
});

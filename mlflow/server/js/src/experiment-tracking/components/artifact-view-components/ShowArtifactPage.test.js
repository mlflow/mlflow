import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactPage from './ShowArtifactPage';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import ShowArtifactMapView from './ShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';
import {
  IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS,
  MAP_EXTENSIONS,
  HTML_EXTENSIONS,
} from '../../../common/utils/FileUtils';
import { RunTag } from '../../sdk/MlflowMessages';

describe('ShowArtifactPage', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      runUuid: 'fakeUuid',
    };
    ShowArtifactPage.prototype.fetchArtifacts = jest.fn();
    commonProps = { ...minimalProps, path: 'fakepath' };
    wrapper = shallow(<ShowArtifactPage {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactPage {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render select-to-preview view when path is unspecified', () => {
    wrapper = shallow(<ShowArtifactPage {...minimalProps} />);
    expect(wrapper.find('.select-preview-outer-container').length).toBe(1);
  });

  test('should render logged model view when path is in runs tag logged model history', () => {
    wrapper.setProps({
      path: 'somePath',
      runTags: {
        'mlflow.log-model.history': RunTag.fromJs({
          key: 'mlflow.log-model.history',
          value: '[{"artifact_path":"somePath","flavors":{"keras":{},"python_function":{}}}]',
        }),
      },
    });
    expect(wrapper.find(ShowArtifactLoggedModelView).length).toBe(1);
  });

  test('should render select-to-preview view when path has no extension', () => {
    wrapper.setProps({ path: 'file_without_extension', runUuid: 'runId' });
    expect(wrapper.find('.select-preview-outer-container').length).toBe(1);
  });

  test('should render select-to-preview view when path has unknown extension', () => {
    wrapper.setProps({ path: 'file.unknown', runUuid: 'runId' });
    expect(wrapper.find('.select-preview-outer-container').length).toBe(1);
  });

  test('should render image view for common image extensions', () => {
    IMAGE_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(ShowArtifactImageView).length).toBe(1);
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
      expect(wrapper.find(ShowArtifactMapView).length).toBe(1);
    });
  });

  test('should render text view for common text extensions', () => {
    TEXT_EXTENSIONS.forEach((ext) => {
      wrapper.setProps({ path: `image.${ext}`, runUuid: 'runId' });
      expect(wrapper.find(ShowArtifactTextView).length).toBe(1);
    });
  });
});

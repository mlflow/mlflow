/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactMapView from './ShowArtifactMapView';
import { mountWithIntl } from '../../../common/utils/TestUtils.enzyme';

describe('ShowArtifactMapView', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakepath',
      runUuid: 'fakeUuid',
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactMapView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactMapView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactMapView {...props} />);
    setImmediate(() => {
      expect(wrapper.find('.artifact-map-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeUndefined();
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-map-view-loading').length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render simple geoJSON in map view', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      const geojson = {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [125.6, 10.1],
        },
        properties: {
          name: 'Dinagat Islands',
        },
      };
      return Promise.resolve(JSON.stringify(geojson));
    });

    const div = global.document.createElement('div');
    global.document.body.appendChild(div);
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = mountWithIntl(<ShowArtifactMapView {...props} />, {
      attachTo: div,
    });

    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('#map')).toHaveLength(1);
      const center = wrapper.instance().leafletMap.getCenter();
      expect(center.lat).toBeCloseTo(10.1);
      expect(center.lng).toBeCloseTo(125.6);
      done();
    });
  });

  test('should fetch artifacts on component update', () => {
    instance = wrapper.instance();
    instance.fetchArtifacts = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchArtifacts).toHaveBeenCalled();
    expect(instance.props.getArtifact).toHaveBeenCalled();
  });
});

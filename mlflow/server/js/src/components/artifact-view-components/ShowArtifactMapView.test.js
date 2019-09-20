import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactMapView from './ShowArtifactMapView';

describe('Render test', () => {
  it('renders empty map container', () => {
    const wrapper = shallow(<ShowArtifactMapView path='fakepath' runUuid='fakerunuuid' />);

    wrapper.setState({ loading: false });

    expect(wrapper.find('.map-container')).toHaveLength(1);
    expect(wrapper.find('#map')).toHaveLength(1);
  });

  it('shows simple geoJSON', () => {
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

    const div = global.document.createElement('div');
    global.document.body.appendChild(div);

    const wrapper = mount(<ShowArtifactMapView path='fakepath' runUuid='fakerunuuid' />, {
      attachTo: div,
    });

    wrapper.setState({ loading: false });
    wrapper.update();
    wrapper.setState({ features: geojson });
    const center = wrapper.instance().leafletMap.getCenter();
    expect(center.lat).toBeCloseTo(10.1);
    expect(center.lng).toBeCloseTo(125.6);
  });
});

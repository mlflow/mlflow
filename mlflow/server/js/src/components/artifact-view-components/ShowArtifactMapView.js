import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactMapView.css';
import { getRequestHeaders } from '../../setupAjaxHeaders';
// There is problem with the marker icon so we loaded the latest css directly from unpkg
// import 'leaflet/dist/leaflet.css';
import L from 'leaflet';


class ShowArtifactMapView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);

    };

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
  };

  state = {
    loading: true,
    error: undefined,
    features: undefined,
  };

  componentWillMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }

    if (window.map !== undefined) {
      if (window.map.hasOwnProperty('_layers')) {
        window.map.off();
        window.map.remove();
        document.getElementsByClassName('map-container')[0].innerHTML = "<div id='map'></div>";
        window.map = undefined;
      };
    };

    const map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    function onEachFeature(feature, layer) {
			if (feature.properties && feature.properties.popupContent) {
				var popupContent = feature.properties.popupContent;
				layer.bindPopup(popupContent);
			};
		};

    var geojsonLayer = L.geoJSON(this.state.features, {
			onEachFeature: onEachFeature
		}).addTo(map);
    map.fitBounds(geojsonLayer.getBounds());
    window.map = map;
  }

  render() {
    if (this.state.loading) {
      return (
        <div>
          Loading...
        </div>
      );
    }
    if (this.state.error) {
      return (
        <div>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      return (
        <div className="map-container">
          <div id="map"></div>
        </div>
      );
    }
  }

  fetchArtifacts() {
    const getArtifactRequest = new Request(getSrc(this.props.path, this.props.runUuid), {
      method: 'GET',
      redirect: 'follow',
      headers: new Headers(getRequestHeaders(document.cookie))
    });
    fetch(getArtifactRequest).then((response) => {
      return response.blob();
    }).then((blob) => {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        this.setState({ features: JSON.parse(event.target.result), loading: false });
      };
      fileReader.readAsText(blob);
    });
  }
}

export default ShowArtifactMapView;

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactMapView.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconRetina from 'leaflet/dist/images/marker-icon-2x.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

function onEachFeature(feature, layer) {
  if (feature.properties && feature.properties.popupContent) {
    const { popupContent } = feature.properties;
    layer.bindPopup(popupContent);
  }
}

class ShowArtifactMapView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
    this.leafletMap = undefined;
    this.mapDivId = 'map';
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
    getArtifact: PropTypes.func,
  };

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  state = {
    loading: true,
    error: undefined,
    features: undefined,
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }

    if (this.leafletMap !== undefined) {
      if (this.leafletMap.hasOwnProperty('_layers')) {
        this.leafletMap.off();
        this.leafletMap.remove();
        const inner = "<div id='" + this.mapDivId + "'></div>";
        document.getElementsByClassName('map-container')[0].innerHTML = inner;
        this.leafletMap = undefined;
      }
    }

    if (this.state.features !== undefined) {
      const map = L.map(this.mapDivId);

      // Load tiles from OSM with the corresponding attribution
      // Potentially, these could be set in an ENV VAR to use a custom map
      const tilesURL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
      const attr = '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors';

      L.tileLayer(tilesURL, {
        attribution: attr,
      }).addTo(map);

      const geojsonLayer = L.geoJSON(this.state.features, {
        style(feature) {
          return feature.properties && feature.properties.style;
        },
        pointToLayer(feature, latlng) {
          if (feature.properties && feature.properties.style) {
            return L.circleMarker(latlng, feature.properties && feature.properties.style);
          } else if (feature.properties && feature.properties.icon) {
            return L.marker(latlng, {
              icon: L.icon(feature.properties && feature.properties.icon),
            });
          }
          return L.marker(latlng, {
            icon: L.icon({
              iconRetinaUrl: iconRetina,
              iconUrl: icon,
              shadowUrl: iconShadow,
              iconSize: [24, 36],
              iconAnchor: [12, 36],
            }),
          });
        },
        onEachFeature: onEachFeature,
      }).addTo(map);
      map.fitBounds(geojsonLayer.getBounds());
      this.leafletMap = map;
    }
  }

  render() {
    if (this.state.loading) {
      return <div className='artifact-map-view-loading'>Loading...</div>;
    }
    if (this.state.error) {
      return (
        <div className='artifact-map-view-error'>
          Oops, we couldn't load your file because of an error.
        </div>
      );
    } else {
      return (
        <div className='map-container'>
          <div id={this.mapDivId}></div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((rawFeatures) => {
        const parsedFeatures = JSON.parse(rawFeatures);
        this.setState({ features: parsedFeatures, loading: false });
      })
      .catch((error) => {
        this.setState({ error: error, loading: false, features: undefined });
      });
  }
}

export default ShowArtifactMapView;

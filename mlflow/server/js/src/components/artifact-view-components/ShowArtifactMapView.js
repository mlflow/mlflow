import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactMapView.css';
import { getRequestHeaders } from '../../setupAjaxHeaders';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconRetina from 'leaflet/dist/images/marker-icon-2x.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

function onEachFeature(feature, layer) {
  if (feature.properties && feature.properties.popupContent) {
    const popupContent = feature.properties.popupContent;
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
      return <div>Loading...</div>;
    }
    if (this.state.error) {
      return <div>Oops, we couldn't load your file because of an error.</div>;
    } else {
      return (
        <div className='map-container'>
          <div id={this.mapDivId}></div>
        </div>
      );
    }
  }

  fetchArtifacts() {
    const getArtifactRequest = new Request(getSrc(this.props.path, this.props.runUuid), {
      method: 'GET',
      redirect: 'follow',
      headers: new Headers(getRequestHeaders(document.cookie)),
    });
    fetch(getArtifactRequest)
      .then((response) => {
        return response.blob();
      })
      .then((blob) => {
        const fileReader = new FileReader();
        fileReader.onload = (event) => {
          try {
            this.setState({ features: JSON.parse(event.target.result), loading: false });
          } catch (error) {
            console.error(error);
            this.setState({ error, loading: false, features: undefined });
          }
        };
        fileReader.readAsText(blob);
      })
      .catch((error) => {
        console.error(error);
        this.setState({ error, loading: false, features: undefined });
      });
  }
}

export default ShowArtifactMapView;

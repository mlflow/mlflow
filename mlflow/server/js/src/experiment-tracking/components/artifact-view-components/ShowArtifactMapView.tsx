/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { getArtifactContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactMapView.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
// @ts-expect-error TS(2307): Cannot find module 'leaflet/dist/images/marker-ico... Remove this comment to see the full error message
import icon from 'leaflet/dist/images/marker-icon.png';
// @ts-expect-error TS(2307): Cannot find module 'leaflet/dist/images/marker-ico... Remove this comment to see the full error message
import iconRetina from 'leaflet/dist/images/marker-icon-2x.png';
// @ts-expect-error TS(2307): Cannot find module 'leaflet/dist/images/marker-sha... Remove this comment to see the full error message
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

function onEachFeature(feature: any, layer: any) {
  if (feature.properties && feature.properties.popupContent) {
    const { popupContent } = feature.properties;
    layer.bindPopup(popupContent);
  }
}

type OwnProps = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
};

type State = any;

type Props = OwnProps & typeof ShowArtifactMapView.defaultProps;

class ShowArtifactMapView extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
    this.leafletMap = undefined;
    this.mapDivId = 'map';
  }

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  leafletMap: any;
  mapDivId: any;

  state = {
    loading: true,
    error: undefined,
    features: undefined,
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps: Props) {
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
        style(feature: any) {
          return feature.properties && feature.properties.style;
        },
        pointToLayer(feature: any, latlng: any) {
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
    const artifactLocation = getArtifactLocationUrl(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((rawFeatures: any) => {
        const parsedFeatures = JSON.parse(rawFeatures);
        this.setState({ features: parsedFeatures, loading: false });
      })
      .catch((error: any) => {
        this.setState({ error: error, loading: false, features: undefined });
      });
  }
}

export default ShowArtifactMapView;

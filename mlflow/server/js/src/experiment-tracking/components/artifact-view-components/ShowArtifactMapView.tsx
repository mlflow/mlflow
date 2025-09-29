/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactMapView.css';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconRetina from 'leaflet/dist/images/marker-icon-2x.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified, type FetchArtifactUnifiedFn } from './utils/fetchArtifactUnified';

function onEachFeature(feature: any, layer: any) {
  if (feature.properties && feature.properties.popupContent) {
    const { popupContent } = feature.properties;
    layer.bindPopup(popupContent);
  }
}

type Props = {
  runUuid: string;
  path: string;
  getArtifact: FetchArtifactUnifiedFn;
} & LoggedModelArtifactViewerProps;

type State = any;

class ShowArtifactMapView extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
    this.leafletMap = undefined;
    this.mapDivId = 'map';
  }

  static defaultProps = {
    getArtifact: fetchArtifactUnified,
  };

  leafletMap: any;
  mapDivId: any;
  mapRef: HTMLDivElement | null = null;

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

    if (this.state.features !== undefined && this.mapRef) {
      const map = L.map(this.mapRef);

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
      return <ArtifactViewSkeleton className="artifact-map-view-loading" />;
    }
    if (this.state.error) {
      return <ArtifactViewErrorState className="artifact-map-view-error" />;
    } else {
      return (
        <div className="map-container">
          <div
            id={this.mapDivId}
            ref={(ref) => {
              this.mapRef = ref;
            }}
          />
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const { path, runUuid, isLoggedModelsMode, loggedModelId, experimentId, entityTags } = this.props;

    this.props
      .getArtifact?.({ path, runUuid, isLoggedModelsMode, loggedModelId, experimentId, entityTags }, getArtifactContent)
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

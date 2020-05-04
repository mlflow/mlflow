import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './ShowArtifactImageView.css';
import { getSrc } from './ShowArtifactPage';
import Plot from 'react-plotly.js';
import Utils from '../../../common/utils/Utils';

class ShowArtifactImageView extends Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: true,
      width: 0,
      height: 0,
      dataURL: '',
    };
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
  };

  componentDidMount = () => {
    // For a gif image, we don't have to do anything here because img tag fetches the image.
    if (this.isGif()) {
      return;
    }

    // For a static image, call fetchImage to load the image and convert it to data URI for plotly.
    this.fetchImage();
  };

  componentDidUpdate = (prevProps) => {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      if (this.isGif()) {
        return;
      }

      this.fetchImage();
    }
  };

  getSrc = () => {
    const { path, runUuid } = this.props;
    return getSrc(path, runUuid);
  };

  isGif = () => {
    return this.props.path.endsWith('.gif');
  };

  fetchImage = () => {
    this.setState({ loading: true });
    const img = new Image();
    img.setAttribute('crossOrigin', 'anonymous');
    img.onload = () => {
      const { width, height } = img;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const dataURL = canvas.toDataURL('image/png');
      this.setState({ loading: false, dataURL, width, height });
    };
    img.src = this.getSrc();
  };

  renderGif = () => {
    const { loading } = this.state;
    return (
      <React.Fragment>
        <div style={{ display: loading ? 'block' : 'none' }}>Loading...</div>
        <img
          src={this.getSrc()}
          alt={Utils.baseName(this.props.path)}
          onLoadStart={() => this.setState({ loading: true })}
          onLoad={() => this.setState({ loading: false })}
          style={{ height: '100%', display: loading ? 'none' : 'block' }}
        />
      </React.Fragment>
    );
  };

  renderStaticImage = () => {
    const { loading, dataURL, width, height } = this.state;

    if (loading) {
      return <div className='artifact-image-view-loading'>Loading...</div>;
    }
    return (
      <Plot
        layout={{
          width: width * (500 / height),
          height: 500,
          xaxis: { visible: false, range: [0, width] },
          yaxis: { visible: false, range: [0, height], scaleanchor: 'x', scaleratio: 1 },
          images: [
            {
              source: dataURL,
              xref: 'x',
              yref: 'y',
              x: 0,
              y: 0,
              xanchor: 'left',
              yanchor: 'bottom',
              sizex: width,
              sizey: height,
            },
          ],
          margin: { l: 0, r: 0, t: 0, b: 0 },
        }}
        config={{
          displaylogo: false,
          scrollZoom: true,
          doubleClick: 'reset',
          modeBarButtonsToRemove: [
            'hoverCompareCartesian',
            'hoverClosestCartesian',
            'lasso2d',
            'sendDataToCloud',
            'select2d',
            'toggleSpikelines',
          ],
        }}
        useResizeHandler
      />
    );
  };

  render() {
    return (
      <div className='image-outer-container'>
        <div className='image-container'>
          {this.isGif() ? this.renderGif() : this.renderStaticImage()}
        </div>
      </div>
    );
  }
}

export default ShowArtifactImageView;

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './ShowArtifactImageView.css';
import { getSrc } from './ShowArtifactPage';
import Plot from 'react-plotly.js';

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
    this.fetchImage();
  };

  componentDidUpdate = prevProps => {
    if (prevProps.path !== this.props.path) {
      this.fetchImage();
    }
  };

  getSrc = () => {
    const { path, runUuid } = this.props;
    return getSrc(path, runUuid);
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

  render() {
    const { loading, dataURL, width, height } = this.state;

    if (loading) return <div>Loading...</div>;

    return (
      <div className="image-outer-container">
        <div className="image-container">
          {
            <Plot
              data={[
                {
                  x: [0, width],
                  y: [0, height],
                  type: 'scatter',
                  mode: 'markers',
                  marker: { opacity: 0, size: 0 },
                  hoverinfo: 'none',
                },
              ]}
              layout={{
                width: width * (500 / height),
                height: 500,
                xaxis: { visible: false, autorange: true },
                yaxis: { visible: false, autorange: true, scaleanchor: 'x', scaleratio: 1 },
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
          }
        </div>
      </div>
    );
  }
}

export default ShowArtifactImageView;

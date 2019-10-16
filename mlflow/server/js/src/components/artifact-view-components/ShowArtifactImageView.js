import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './ShowArtifactImageView.css';
import { getSrc } from './ShowArtifactPage';
import Plot from 'react-plotly.js';

class ShowArtifactImageView extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
  };

  constructor(props) {
    super(props);
    this.state = {
      width: 0,
      height: 0,
    };
  }

  getSrc = () => {
    const { path, runUuid } = this.props;
    return getSrc(path, runUuid);
  };

  resize = () => {
    const img = new Image();
    img.src = this.getSrc();
    img.onload = () => this.setState({ width: img.width, height: img.height });
  };

  componentDidMount = () => {
    this.resize();
  };

  componentDidUpdate = prevProps => {
    if (prevProps.path !== this.props.path) {
      this.resize();
    }
  };

  render() {
    const { width, height } = this.state;

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
                    source: this.getSrc(),
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

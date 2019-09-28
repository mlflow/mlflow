import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import {
  getBasename, getExtension, IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS, MAP_EXTENSIONS
} from '../utils/FileUtils';
import { getArtifactRootUri, getArtifacts } from '../reducers/Reducers';
import { ArtifactNode as ArtifactUtils, ArtifactNode } from '../utils/ArtifactUtils';
import { decorators, Treebeard } from 'react-treebeard';
import bytes from 'bytes';
import './ArtifactView.css';
import ShowArtifactPage, {getSrc} from './artifact-view-components/ShowArtifactPage';
import spinner from '../static/mlflow-spinner.png';

class ArtifactView extends Component {
  constructor(props) {
    super(props);
    this.onToggleTreebeard = this.onToggleTreebeard.bind(this);
    this.getTreebeardData = this.getTreebeardData.bind(this);
    this.getRealPath = this.getRealPath.bind(this);
    this.shouldShowTreebeard = this.shouldShowTreebeard.bind(this);
    this.activeNodeIsDirectory = this.activeNodeIsDirectory.bind(this);
  }
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    // The root artifact node.
    artifactNode: PropTypes.instanceOf(ArtifactNode).isRequired,
    fetchArtifacts: PropTypes.func.isRequired,
    artifactRootUri: PropTypes.string.isRequired,
  };

  state = {
    activeNodeId: undefined,
    toggledNodeIds: {},
    requestedNodeIds: new Set(),
  };

  render() {
    // If not hydrated then try to get the data before rendering this view.
    return (
      <div>
        {this.shouldShowTreebeard() ?
          <div className="artifact-view">
            <div className="artifact-left">
              <Treebeard
                data={this.getTreebeardData(this.props.artifactNode)}
                onToggle={this.onToggleTreebeard}
                style={TREEBEARD_STYLE}
                decorators={decorators}
              />
            </div>
            <div className="artifact-right">
              <div className="artifact-info">
                {this.state.activeNodeId ?
                  <div>
                    {!this.activeNodeIsDirectory() ?
                      <div className="artifact-info-link">
                        <a href={getSrc(this.state.activeNodeId, this.props.runUuid)}
                           target="_blank"
                           title="Download artifact">
                          <i className="fas fa-download"/>
                        </a>
                      </div>
                      :
                      null
                    }
                    <div className="artifact-info-path">
                      <label>Full Path:</label> {this.getRealPath()}
                    </div>
                    <div className="artifact-info-size">
                      <label>Size:</label> {this.getSize()}
                    </div>
                  </div>
                  :
                  null
                }
              </div>
              <ShowArtifactPage runUuid={this.props.runUuid} path={this.state.activeNodeId}/>
            </div>
            <div className="artifact-center">
            </div>
          </div>
          :
          <div className="empty-artifact-outer-container">
            <div className="empty-artifact-container">
              <div>
                {/* TODO: put a nice image here */}
              </div>
              <div>
                <div className="no-artifacts">No Artifacts Recorded</div>
                <div className="no-artifacts-info">
                  Use the log artifact APIs to store file outputs from MLflow runs.
                </div>
              </div>
            </div>
          </div>
        }
      </div>
    );
  }

  onToggleTreebeard(dataNode, toggled) {
    const { id, loading } = dataNode;
    const newRequestedNodeIds = new Set(this.state.requestedNodeIds);
    // - loading indicates that this node is a directory and has not been loaded yet.
    // - requestedNodeIds keeps track of in flight requests.
    if (loading && !this.state.requestedNodeIds.has(id)) {
      this.props.fetchArtifacts(this.props.runUuid, id);
    }
    this.setState({
      activeNodeId: id,
      toggledNodeIds: {
        ...this.state.toggledNodeIds,
        [id]: toggled,
      },
      requestedNodeIds: newRequestedNodeIds,
    });
  }

  shouldShowTreebeard() {
    return !ArtifactUtils.isEmpty(this.props.artifactNode);
  }

  getTreebeardData(artifactNode) {
    const isRoot = artifactNode.isRoot;
    if (isRoot) {
      if (artifactNode.children) {
        return Object.values(artifactNode.children).map((c) => this.getTreebeardData(c));
      }
      // This case should never happen since we should never call this function on an empty root.
      throw Error("unreachable code.");
    }

    let id;
    let name;
    let toggled;
    let children;
    let active;

    if (artifactNode.fileInfo) {
      const path = artifactNode.fileInfo.path;
      id = path;
      name = getBasename(path);
    }

    const toggleState = this.state.toggledNodeIds[id];
    if (toggleState) {
      toggled = toggleState;
    }

    if (artifactNode.children) {
      children = Object.values(artifactNode.children).map((c) => this.getTreebeardData(c));
    }

    if (this.state.activeNodeId === id) {
      active = true;
    }

    const loading = artifactNode.children !== undefined && !artifactNode.isLoaded;

    return {
      id,
      name,
      toggled,
      children,
      active,
      loading,
    };
  }

  getRealPath() {
    if (this.state.activeNodeId) {
      return `${this.props.artifactRootUri}/${this.state.activeNodeId}`;
    }
    return this.props.artifactRootUri;
  }

  getSize() {
    if (this.state.activeNodeId) {
      const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
      const size = node.fileInfo.file_size || '0';
      return bytes(parseInt(size, 10));
    }
    return bytes(0);
  }

  activeNodeIsDirectory() {
    if (this.state.activeNodeId) {
      const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
      return node.fileInfo.is_dir;
    } else {
      // No node is highlighted so we're displaying the root, which is a directory.
      return true;
    }
  }
}


const mapStateToProps = (state, ownProps) => {
  const { runUuid } = ownProps;
  const artifactNode = getArtifacts(runUuid, state);
  const artifactRootUri = getArtifactRootUri(runUuid, state);
  return { artifactNode, artifactRootUri };
};

export default connect(mapStateToProps)(ArtifactView);

const TREEBEARD_STYLE = {
  tree: {
    base: {
      listStyle: 'none',
      margin: 0,
      padding: 0,
      backgroundColor: '#FAFAFA',
      fontSize: '14px',
      maxWidth: '500px',
      height: '556px',
      overflow: 'scroll',
    },
    node: {
      base: {
        position: 'relative'
      },
      link: {
        cursor: 'pointer',
        position: 'relative',
        padding: '0px 5px',
        display: 'block'
      },
      activeLink: {
        background: '#c7c2d0'
      },
      toggle: {
        base: {
          position: 'relative',
          display: 'inline-block',
          verticalAlign: 'top',
          marginLeft: '-5px',
          height: '24px',
          width: '24px'
        },
        wrapper: {
          position: 'absolute',
          top: '50%',
          left: '50%',
          margin: '-7px 0 0 -7px',
          height: '14px'
        },
        height: 14,
        width: 14,
        arrow: {
          fill: '#7a7a7a',
          strokeWidth: 0
        }
      },
      header: {
        base: {
          display: 'inline-block',
          verticalAlign: 'top',
          color: '#333'
        },
        connector: {
          width: '2px',
          height: '12px',
          borderLeft: 'solid 2px black',
          borderBottom: 'solid 2px black',
          position: 'absolute',
          top: '0px',
          left: '-21px'
        },
        title: {
          lineHeight: '24px',
          verticalAlign: 'middle'
        }
      },
      subtree: {
        listStyle: 'none',
        paddingLeft: '19px'
      },
    }
  }
};

// eslint-disable-next-line react/prop-types
decorators.Header = ({style, node}) => {
  let iconType;
  if (node.children) {
    iconType = 'folder';
  } else {
    const extension = getExtension(node.name);
    if (IMAGE_EXTENSIONS.has(extension)) {
      iconType = 'file-image';
    } else if (TEXT_EXTENSIONS.has(extension)) {
      iconType = 'file-code';
    } else if (MAP_EXTENSIONS.has(extension)) {
      iconType = 'map-marker-alt';
    } else {
      iconType = 'file-alt';
    }
  }
  const iconClass = `fa fa-${iconType}`;
  const iconStyle = {marginRight: '5px'};

  return (
    <div style={style.base}>
      <div style={style.title}>
        <i className={iconClass} style={iconStyle}/>

        {node.name}
      </div>
    </div>
  );
};

// eslint-disable-next-line react/prop-types
decorators.Loading = ({style}) => {
  return (
    <div style={style}>
      <img alt="" className="loading-spinner" src={spinner}/>
      {' '}loading...
    </div>
  );
};

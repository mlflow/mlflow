import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { Popover } from 'antd';
import Routes from '../Routes';

class RunLinksPopover extends React.Component {
  static propTypes = {
    experimentId: PropTypes.string.isRequired,
    onCloseClick: PropTypes.func.isRequired,
  };

  constructor(props) {
    super(props);
    this.state = {
      visible: false,
      x: 0,
      y: 0,
      runsData: [],
    };
  }

  renderContent = () => {
    const { runsData } = this.state;
    const { experimentId } = this.props;
    return (
      <div>
        {runsData.map(({ name, runUuid, color }, index) => (
          <Link key={`${runUuid}-${index}`} to={Routes.getRunPageRoute(experimentId, runUuid)}>
            <p style={{ color }}>{name}</p>
          </Link>
        ))}
      </div>
    );
  };

  renderTitle = () => {
    return (
      <div>
        <span>Jump to the run</span>
        <a onClick={() => this.setState({ visible: false })} style={{ float: 'right' }}>
          <i className="fas fa-times"></i>
        </a>
      </div>
    );
  };

  render() {
    const { visible, x, y } = this.state;
    return (
      <Popover
        content={this.renderContent()}
        title={this.renderTitle()}
        placement="top"
        visible={visible}
      >
        {/* dummy div to control the position of the popover */}
        <div
          style={{
            left: x,
            top: y,
            position: 'absolute',
          }}
        />
      </Popover>
    );
  }
}

export default RunLinksPopover;

import React from 'react';
import PropTypes from 'prop-types';
import { Popover } from 'antd';

class RunLinksPopover extends React.Component {
  static propTypes = {
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired,
    visible: PropTypes.bool.isRequired,
    onCloseClick: PropTypes.func.isRequired,
    runsData: PropTypes.arrayOf(Object).isRequired,
  };

  renderPopoverContent = () => {
    const { runsData } = this.props;
    return (
      <div>
        {runsData.map(({ runUuid, color }) => (
          <p key={runUuid}>
            <a href="" style={{ color }}>
              {runUuid}
            </a>
          </p>
        ))}
      </div>
    );
  };

  renderTitle = () => {
    const { onCloseClick } = this.props;
    return (
      <div>
        <span>Jump to the run</span>
        <a onClick={onCloseClick} style={{ float: 'right' }}>
          <i className="fas fa-times"></i>
        </a>
      </div>
    );
  };

  render() {
    const { x, y, visible } = this.props;
    return (
      <Popover
        content={this.renderPopoverContent()}
        title={this.renderTitle()}
        placement="topLeft"
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

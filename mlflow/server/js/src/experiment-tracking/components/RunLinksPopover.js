import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { Popover } from 'antd';
import Routes from '../routes';
import { IconButton } from '../../common/components/IconButton';
import Utils from '../../common/utils/Utils';

export class RunLinksPopover extends React.Component {
  static propTypes = {
    experimentId: PropTypes.string.isRequired,
    visible: PropTypes.bool.isRequired,
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired,
    runItems: PropTypes.arrayOf(PropTypes.object).isRequired,
    handleClose: PropTypes.func.isRequired,
    handleKeyDown: PropTypes.func.isRequired,
    handleVisibleChange: PropTypes.func.isRequired,
  };

  componentDidMount() {
    document.addEventListener('keydown', this.props.handleKeyDown);
  }

  componentWillUnmount() {
    document.removeEventListener('keydown', this.props.handleKeyDown);
  }

  renderContent = () => {
    const { experimentId, runItems } = this.props;
    return (
      <div>
        {runItems.map(({ name, runId, color, y }, index) => {
          const key = `${runId}-${index}`;
          const to = Routes.getRunPageRoute(experimentId, runId);
          return (
            <Link key={key} to={to}>
              <p style={{ color }}>
                <i className='fas fa-external-link-alt' style={{ marginRight: 5 }} />
                {`${name}, ${Utils.formatMetric(y)}`}
              </p>
            </Link>
          );
        })}
      </div>
    );
  };

  renderTitle = () => {
    const { handleClose } = this.props;
    return (
      <div>
        <span>Jump to individual runs</span>
        <IconButton
          icon={<i className='fas fa-times' />}
          onClick={handleClose}
          style={{ float: 'right', marginLeft: '7px' }}
        />
      </div>
    );
  };

  render() {
    const { visible, x, y, handleVisibleChange } = this.props;
    return (
      <Popover
        content={this.renderContent()}
        title={this.renderTitle()}
        placement='left'
        visible={visible}
        onVisibleChange={handleVisibleChange}
      >
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

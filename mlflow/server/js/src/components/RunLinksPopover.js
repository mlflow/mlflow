import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { Popover } from 'antd';
import Routes from '../Routes';

export class RunLinksPopover extends React.Component {
  static propTypes = {
    experimentId: PropTypes.number.isRequired,
  };

  constructor(props) {
    super(props);
    this.state = {
      visible: false,
      x: 0,
      y: 0,
      runItems: [],
    };
  }

  componentDidMount() {
    document.addEventListener('keydown', this.handleKeyDown);
  }

  componentWillUnmount() {
    document.removeEventListener('keydown', this.handleKeyDown);
  }

  updateState = (visible, x, y, runItems) => this.setState({visible, x, y, runItems});

  hide = () => this.setState({ visible: false });

  handleVisibleChange = (visible) => this.setState({ visible });

  handleKeyDown = ({ key }) => {
    if (key === 'Escape') {
      this.hide();
    }
  };

  renderContent = () => {
    const { runItems } = this.state;
    const { experimentId } = this.props;
    return (
      <div>
        {runItems.map(({ name, runUuid, color }, index) => {
          const key = `${runUuid}-${index}`;
          const to = Routes.getRunPageRoute(experimentId, runUuid);
          return (
            <Link key={key} to={to}>
              <p style={{ color }}>
                <i className="fas fa-external-link-alt" style={{ marginRight: 5 }} />
                {name}
              </p>
            </Link>
          );
        })}
      </div>
    );
  };

  renderTitle = () => {
    return (
      <div>
        <span>Jump to the run</span>
        <a onClick={this.hide} style={{ float: 'right' }}>
          <i className="fas fa-times" />
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
        onVisibleChange={this.handleVisibleChange}
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

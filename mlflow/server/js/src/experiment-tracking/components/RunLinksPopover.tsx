import React from 'react';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { IconButton } from '../../common/components/IconButton';
import Utils from '../../common/utils/Utils';
import { LegacyPopover } from '@databricks/design-system';
import { type RunItem } from '../types';

type Props = {
  experimentId: string;
  visible: boolean;
  x: number;
  y: number;
  runItems: RunItem[];
  handleClose: () => void;
  handleKeyDown: ({ key }: { key: string }) => void;
  handleVisibleChange: (visible: boolean) => void;
};

export class RunLinksPopover extends React.Component<Props> {
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
                <i className="fa fa-external-link-o" style={{ marginRight: 5 }} />
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
          icon={<i className="fa fa-times" />}
          onClick={handleClose}
          style={{ float: 'right', marginLeft: '7px' }}
        />
      </div>
    );
  };

  render() {
    const { visible, x, y, handleVisibleChange } = this.props;
    return (
      <LegacyPopover
        content={this.renderContent()}
        title={this.renderTitle()}
        placement="left"
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
      </LegacyPopover>
    );
  }
}

import React from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';

const spacingSizes = [4, 8, 16, 24, 32, 40];

export const getMarginSize = (size) => {
  switch (size) {
    case 'small':
      return 4;
    case 'medium':
    case undefined:
      return 8;
    case 'large':
      return 16;
    default:
      return spacingSizes[size];
  }
};

/**
 * Spaces its children according to the direction and size specified.
 * @param props size: One of "small", "medium" or "large". Default small.
 * @param props direction: One of "horizontal" or "vertical". Default vertical.
 */
export class Spacer extends React.Component {
  static propTypes = {
    children: PropTypes.node,
    size: PropTypes.oneOf([
      undefined, 'small', 'medium', 'large', 0, 1, 2, 3, 4, 5
    ]),
    direction: PropTypes.string,
  }
  render() {
    const { children, size = 'small', direction = 'vertical' } = this.props;
    const marginSize = getMarginSize(size);
    const style = css(styles(marginSize, direction));
    return (<div className={style}>{children}</div>);
  }
}

const styles = (marginSize, direction) =>
  (direction === 'horizontal' ? {
      display: 'flex',
      alignItems: 'center',
      '> :not(:last-child)': { marginRight: marginSize },
    } :
    {
      '> :not(:last-child)': { marginBottom: marginSize },
    });

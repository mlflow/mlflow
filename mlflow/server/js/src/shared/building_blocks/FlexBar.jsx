import React from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';

/**
 * A component used to arrange sub-components horizontally, with some justified to the
 * left and some to the right.
 * @param props left: component to render aligned to the left.
 * @param props right: component to render aligned to the right.
 */
export class FlexBar extends React.Component {
  static propTypes = {
    left: PropTypes.node,
    right: PropTypes.node,
  }

  render() {
    const { left, right } = this.props;
    return (
      <div className={css(styles.flexBox)}>
        {left}
        {right}
      </div>
    );
  }
}

const styles = {
  flexBox: {
    display: 'flex',
    justifyContent: 'space-between',
    flexFlow: 'row wrap',
    gap: '12px',
  },
};

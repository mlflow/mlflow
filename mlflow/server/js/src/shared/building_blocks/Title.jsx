import React from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';
import { headerText } from '../colors';

/**
 * A page title component.
 */
export class Title extends React.Component {
  static propTypes = {
    children: PropTypes.node.isRequired,
  };

  render() {
    const { children } = this.props;
    return <h1 className={css(styles.header)}>{children}</h1>;
  }
}

const styles = {
  header: {
    margin: 0,
    lineHeight: 1,
    color: headerText,
    fontSize: 21,
    fontWeight: 500,
  },
};

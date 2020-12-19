import React from 'react';
import { css, cx } from 'emotion';
import { gray100 } from '../styles/color';
import PropTypes from 'prop-types';

const PreviewIcon = (props) => (
  <span className={cx(previewClassName, props.className)}>Preview</span>
);

PreviewIcon.propTypes = {
  className: PropTypes.string,
};

const previewClassName = css({
  position: 'relative',
  verticalAlign: 'super',
  lineHeight: 0,
  borderRadius: 7,
  background: gray100,
  fontSize: 11,
  paddingTop: 2,
  paddingBottom: 2,
  paddingLeft: 7,
  paddingRight: 7,
  marginLeft: 3,
  fontWeight: 'normal',
});

export default PreviewIcon;

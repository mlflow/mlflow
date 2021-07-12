import React from 'react';
import { css, cx } from 'emotion';
import PropTypes from 'prop-types';
import { FormattedMessage } from 'react-intl';

export const PreviewIcon = (props) => (
  <span className={cx(previewClassName, props.className)}>
    <FormattedMessage
      defaultMessage='Preview'
      description='Preview badge shown for features which are under preview'
    />
  </span>
);

PreviewIcon.propTypes = {
  className: PropTypes.string,
};

const previewClassName = css({
  display: 'inline-block',
  fontSize: 12,
  lineHeight: '16px',
  fontWeight: 500,
  color: '#2e3840', // Color needs alignment -- not part of any spectrum.
  backgroundColor: '#f3f5f6',
  borderRadius: 16,
  padding: '4px 12px',
});

import React from 'react';
import PropTypes from 'prop-types';

export const TrimmedText = ({ text, maxSize, className }) => {
  if (text.length > maxSize) {
    return <span className={className}>{text.substr(0, maxSize)}...</span>;
  }
  return <span className={className}>{text}</span>;
};

TrimmedText.propTypes = {
  text: PropTypes.string.isRequired,
  maxSize: PropTypes.number.isRequired,
  className: PropTypes.string.isRequired,
};

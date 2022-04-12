import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { FormattedMessage } from 'react-intl';
import { Button, Tooltip } from '@databricks/design-system';

export const CopyButton = ({ copyText }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const handleClick = () => {
    navigator.clipboard.writeText(copyText);
    setShowTooltip(true);
    setTimeout(() => {
      setShowTooltip(false);
    }, 3000);
  };

  const handleMouseLeave = () => {
    setShowTooltip(false);
  };

  return (
    <Tooltip
      title={
        <FormattedMessage
          defaultMessage='Copied'
          description='Tooltip text shown when copy operation completes'
        />
      }
      dangerouslySetAntdProps={{
        visible: showTooltip,
      }}
    >
      <Button type='primary' onClick={handleClick} onMouseLeave={handleMouseLeave}>
        <FormattedMessage defaultMessage='Copy' description='Button text for copy button' />
      </Button>
    </Tooltip>
  );
};

CopyButton.propTypes = {
  copyText: PropTypes.string.isRequired,
};

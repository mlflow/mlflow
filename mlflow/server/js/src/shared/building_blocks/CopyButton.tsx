import React, { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, Tooltip } from '@databricks/design-system';

type Props = {
  copyText: string;
};

export const CopyButton = ({ copyText }: Props) => {
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

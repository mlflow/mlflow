import React from 'react';

const spacingSizes = [4, 8, 16, 24, 32, 40];

const getMarginSize = (size: SpacerProps['size']): number => {
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

type SpacerProps = {
  size?: 'small' | 'medium' | 'large' | 0 | 1 | 2 | 3 | 4 | 5;
  direction?: 'horizontal' | 'vertical';
};

/**
 * Spaces its children according to the direction and size specified.
 * @param props size: One of "small", "medium", "large", 0, 1, 2, 3, 4, or 5. Default small.
 * @param props direction: One of "horizontal" or "vertical". Default vertical.
 */
export class Spacer extends React.Component<SpacerProps> {
  render() {
    const { children, size = 'small', direction = 'vertical' } = this.props;
    const marginSize = getMarginSize(size);
    const style = styles(marginSize, direction);
    return <div css={style}>{children}</div>;
  }
}

const styles = (marginSize: number, direction: Exclude<SpacerProps['direction'], undefined>) =>
  direction === 'horizontal'
    ? {
        display: 'flex',
        alignItems: 'center',
        '> :not(:last-child)': { marginRight: marginSize },
      }
    : {
        '> :not(:last-child)': { marginBottom: marginSize },
      };

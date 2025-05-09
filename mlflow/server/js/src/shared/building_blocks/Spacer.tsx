/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';

const spacingSizes = [4, 8, 16, 24, 32, 40];

const getMarginSize = (size: any) => {
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
  size?: any; // TODO: PropTypes.oneOf([undefined, 'small', 'medium', 'large', 0, 1, 2, 3, 4, 5])
  direction?: string;
};

/**
 * Spaces its children according to the direction and size specified.
 * @param props size: One of "small", "medium" or "large". Default small.
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

const styles = (marginSize: any, direction: any) =>
  direction === 'horizontal'
    ? {
        display: 'flex',
        alignItems: 'center',
        '> :not(:last-child)': { marginRight: marginSize },
      }
    : {
        '> :not(:last-child)': { marginBottom: marginSize },
      };

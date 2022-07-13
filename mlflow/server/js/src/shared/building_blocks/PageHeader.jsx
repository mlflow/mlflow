import React from 'react';
import PropTypes from 'prop-types';
import { Typography, Spacer as DbSpacer, Dropdown, Menu } from '@databricks/design-system';
import { Breadcrumb } from './antd/Breadcrumb';
import { Button } from './antd/Button';
import { RightChevron } from '../icons/RightChevron';
import { PreviewIcon } from './PreviewIcon';
import { Spacer } from './Spacer';

// Note: this button has a different size from normal AntD buttons.
export { Button as HeaderButton };

export function OverflowMenu({ menu }) {
  const overflowMenu = (
    <Menu>
      {menu.map(({ id, itemName, onClick, href, ...otherProps }) => (
        <Menu.Item key={id} onClick={onClick} href={href} data-test-id={id} {...otherProps}>
          {itemName}
        </Menu.Item>
      ))}
    </Menu>
  );

  return (
    menu.length > 0 && (
      <Dropdown overlay={overflowMenu} trigger={['click']}>
        <Button type='secondary' data-test-id='overflow-menu-trigger'>
          ⋮
        </Button>
      </Dropdown>
    )
  );
}
OverflowMenu.propTypes = {
  menu: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      itemName: PropTypes.node.isRequired,
      onClick: PropTypes.func,
      href: PropTypes.string,
    }),
  ),
};

const { Title } = Typography;

/**
 * A page header that includes a title, optional breadcrumb content, and a divider.
 * @param props title: Title text.
 * @param props breadcrumbs: Array of React nodes rendered as antd breadcrumbs.
 */
export class PageHeader extends React.Component {
  static propTypes = {
    title: PropTypes.node.isRequired,
    breadcrumbs: PropTypes.arrayOf(PropTypes.node),
    preview: PropTypes.bool,
    feedbackForm: PropTypes.string,
    children: PropTypes.node,
  };

  render() {
    const { title, breadcrumbs = [], preview, children } = this.props;
    let feedbackLink = null;
    return (
      <>
        {breadcrumbs.length > 0 && (
          <Breadcrumb
            css={styles.breadcrumbOverride}
            separator={
              <span css={styles.iconWrapper}>
                <RightChevron />
              </span>
            }
          >
            {breadcrumbs.map((item, i) => (
              <Breadcrumb.Item key={i}>{item}</Breadcrumb.Item>
            ))}
          </Breadcrumb>
        )}
        <div css={styles.titleContainer}>
          <div css={styles.title}>
            <Spacer size={1} direction='horizontal'>
              <Title level={2}>{title}</Title>
              {preview && <PreviewIcon />}
              {feedbackLink}
            </Spacer>
          </div>
          <div css={styles.buttonGroup}>
            <Spacer direction='horizontal'>{children}</Spacer>
          </div>
        </div>
        <DbSpacer size='medium' />
      </>
    );
  }
}

// Needs to match button height else there will be variable line-heights
const antButtonHeight = '32px';
const styles = {
  titleContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
  },
  title: {
    display: 'flex',
    alignItems: 'center',
    minHeight: antButtonHeight,
    'h1, h2, h3': { marginTop: 0, marginBottom: 0 },
  },
  buttonGroup: {
    flexShrink: 1,
    display: 'flex',
    alignItems: 'flex-end',
  },
  iconWrapper: {
    display: 'inline-block',
    height: 16,
    verticalAlign: 'text-bottom',
    svg: {
      height: '100%',
    },
  },
  breadcrumbOverride: {
    marginBottom: 8,
    '.ant-breadcrumb-separator': {
      // For whatever reason, the svg we're using adds extra whitespace (more on the left than
      // the right). Overriding the antd default margin to equalize the spacing.
      marginLeft: 0,
      marginRight: 2,
    },
  },
};

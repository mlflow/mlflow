import React from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';
import { Title } from './Title';
import { Dropdown, Menu } from 'antd';
import { Breadcrumb } from './antd/Breadcrumb';
import { Button } from './antd/Button';
import { RightChevron } from '../icons/RightChevron';
import { grayRule } from '../colors';
import { PreviewIcon } from './PreviewIcon';
import { Spacer } from './Spacer';

// Note: this button has a different size from normal AntD buttons.
export function HeaderButton({ onClick, children, ...props }){
  return (
    <Button size='default' onClick={onClick} {...props} >
      {children}
    </Button>
  )
};

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
        {breadcrumbs && (
          <Breadcrumb
            className={css(styles.breadcrumbOverride)}
            separator={
              <span className={css(styles.iconWrapper)}>
                <RightChevron />
              </span>
            }
          >
            {breadcrumbs.map((item, i) => (
              <Breadcrumb.Item key={i}>{item}</Breadcrumb.Item>
            ))}
          </Breadcrumb>
        )}
        <div className={css(styles.titleContainer)}>
          <div className={css(styles.title)}>
            <Spacer size={1} direction='horizontal'>
              <Title level={3}>{title}</Title>
              {preview && <PreviewIcon />}
              {feedbackLink}
            </Spacer>
          </div>
          <div className={css(styles.buttonGroup)}>
            <Spacer direction='horizontal'>{children}</Spacer>
          </div>
        </div>
        <div className={css(styles.hrWrapper)}>
          <hr className={css(styles.hr)} />
        </div>
      </>
    );
  }
}

// Needs to match button height else there will be variable line-heights
const antButtonHeight = '32px';
const styles = {
  titleContainer: {
    marginBottom: 16,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
  },
  title: {
    display: 'flex',
    alignItems: 'center',
    minHeight: antButtonHeight,
  },
  buttonGroup: {
    flexShrink: 1,
    display: 'flex',
    alignItems: 'flex-end',
    button: {
      padding: '6px 12px',
      display: 'flex', 
      alignItems: 'center',
    }
  },
  hr: {
    marginTop: 0, // hr margin comes from bootstrap. Must override.
    marginBottom: 24,
    height: '1px',
    backgroundColor: grayRule,
    border: 'none',
  },
  hrWrapper: {
    margin: 0,
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

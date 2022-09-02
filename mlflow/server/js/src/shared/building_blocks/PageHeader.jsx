import React from 'react';
import PropTypes from 'prop-types';
import {
  Breadcrumb,
  Button,
  Spacer,
  Dropdown,
  Menu,
  Header,
  OverflowIcon,
} from '@databricks/design-system';
import { PreviewIcon } from './PreviewIcon';

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

  return menu.length > 0 ? (
    <Dropdown overlay={overflowMenu} trigger={['click']} placement='bottomLeft' arrow>
      <Button icon={<OverflowIcon />} data-test-id='overflow-menu-trigger' />
    </Dropdown>
  ) : null;
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
    // eslint-disable-next-line prefer-const
    let feedbackLink = null;
    return (
      <>
        <Header
          breadcrumbs={
            <Breadcrumb includeTrailingCaret={false}>
              {breadcrumbs.map((b, i) => (
                <Breadcrumb.Item key={i}>{b}</Breadcrumb.Item>
              ))}
            </Breadcrumb>
          }
          buttons={children}
          title={title}
          titleAddOns={
            <>
              {preview && <PreviewIcon />}
              {feedbackLink}
            </>
          }
        />
        <Spacer />
      </>
    );
  }
}

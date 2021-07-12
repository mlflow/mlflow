import React from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';
import { Title } from './Title';
import { Typography } from './antd/Typography';
import { Breadcrumb } from './antd/Breadcrumb';
import { RightChevron } from '../icons/RightChevron';
import { grayRule } from '../colors';
import { PreviewIcon } from './PreviewIcon';
import { Spacer } from './Spacer';
import { FlexBar } from './FlexBar';

const { Text } = Typography;

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
    copyText: PropTypes.string,
    rightAlignedTitle: PropTypes.node,
  };

  render() {
    const { title, breadcrumbs = [], preview, copyText, rightAlignedTitle } = this.props;
    let feedbackLink = null;
    return (
      <>
        <div className={css(styles.titleContainer)}>
          <FlexBar
            left={
              <Spacer size={1} direction='horizontal'>
                <Title>
                  {title}
                  {copyText && <Text copyable={{ text: copyText }}/>}
                </Title>
                {preview && <PreviewIcon/>}
                {feedbackLink}
              </Spacer>
            }
            right={rightAlignedTitle}
          >
          </FlexBar>
        </div>
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
        <div className={css(styles.hrWrapper)}>
          <hr className={css(styles.hr)} />
        </div>
      </>
    );
  }
}

const styles = {
  titleContainer: {
    marginBottom: 16,
  },
  hr: {
    marginTop: 24, // hr margin comes from bootstrap. Must override.
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
    '.ant-breadcrumb-separator': {
      // For whatever reason, the svg we're using adds extra whitespace (more on the left than
      // the right). Overriding the antd default margin to equalize the spacing.
      marginLeft: 0,
      marginRight: 2,
    },
  },
};

import React from 'react';
import PropTypes from 'prop-types';
import { Button, Icon } from 'antd';

const ButtonGroup = Button.Group;

export class SimplePagination extends React.Component {
  static propTypes = {
    currentPage: PropTypes.number.isRequired,
    isLastPage: PropTypes.bool.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
  };

  render() {
    const { currentPage, isLastPage, onClickNext, onClickPrev } = this.props;
    return (
      <div style={{ textAlign: 'right' }}>
        <ButtonGroup>
          <Button
            disabled={currentPage === 1}
            className='prev-page-btn'
            onClick={onClickPrev}
            size='small'
            type='link'
          >
            <Icon type='left' />
          </Button>
          <span>Page {currentPage}</span>
          <Button
            disabled={isLastPage}
            className='next-page-btn'
            onClick={onClickNext}
            size='small'
            type='link'
          >
            <Icon type='right' />
          </Button>
        </ButtonGroup>
      </div>
    );
  }
}

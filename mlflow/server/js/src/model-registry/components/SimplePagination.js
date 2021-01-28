import React from 'react';
import PropTypes from 'prop-types';
import { Button, Dropdown, Icon, Menu } from 'antd';
import { css } from 'emotion';

const ButtonGroup = Button.Group;

export class SimplePagination extends React.Component {
  static propTypes = {
    currentPage: PropTypes.number.isRequired,
    isLastPage: PropTypes.bool.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
    loading: PropTypes.bool,
    maxResultOptions: PropTypes.array,
    handleSetMaxResult: PropTypes.func,
    getSelectedPerPageSelection: PropTypes.func,
  };

  constructDropdown() {
    return (
      <Menu
        className={`pagination-dropdown ${classNames.paginationDropdownMenuWrapper}`}
        onClick={this.props.handleSetMaxResult}
      >
        {this.props.maxResultOptions.map((num_models) => (
          <Menu.Item key={num_models.toString()} title={num_models.toString()}>
            {num_models}
          </Menu.Item>
        ))}
      </Menu>
    );
  }

  render() {
    const { currentPage, isLastPage, onClickNext, onClickPrev } = this.props;
    return (
      <div className={`pagination-section ${classNames.wrapper}`}>
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
          {this.props.maxResultOptions ? (
            <Dropdown disabled={this.props.loading} overlay={this.constructDropdown()}>
              <Button>
                <span>
                  {this.props.getSelectedPerPageSelection()} / page <Icon type='down' />
                </span>
              </Button>
            </Dropdown>
          ) : null}
        </ButtonGroup>
      </div>
    );
  }
}

const classNames = {
  wrapper: css({
    textAlign: 'right',
    paddingBottom: 30,
  }),
  paginationDropdownMenuWrapper: css({
    '.ant-dropdown-menu-item': {
      textAlign: 'center',
    },
  }),
};

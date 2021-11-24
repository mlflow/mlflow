import React from 'react';
import { css } from 'emotion';
import PropTypes from 'prop-types';
import { Button } from './Button';
import { Spacer } from './Spacer';
import { Input } from './antd/Input';
import { FormattedMessage } from 'react-intl';

/**
 * A search box for searching or filtering lists and tables. Use [data-test-id='search-box'] and
 * [data-test-id='search-button'] for integration testing.
 *
 * This component can be either fully controlled or fully uncontrolled. For a fully controlled usage
 * (i.e. search state is handled by the parent component, which enables, for instance,
 * a "clear filters" button), the value and onChange props must be passed in. For a fully
 * uncontrolled usage (i.e. search state does not need to be controlled by a parent component),
 * the value prop must be left undefined.
 *
 * @param props value: value of the input box. This should be left undefined *if and only if*
 * a fully uncontrolled usage is desired.
 * @param props onChange: function to run on input change. Takes an event.
 * @param props onSearch: function to run on search trigger. Takes an event and a string.
 * @param props hintText: placeholder text to show in the text box before user input.
 */
export class SearchBox extends React.Component {
  static propTypes = {
    value: PropTypes.string,
    onChange: PropTypes.func,
    onSearch: PropTypes.func.isRequired,
    placeholder: PropTypes.string,
  };

  getInputValue(value, state) {
    return value === undefined ? state.searchInput : value;
  }

  constructor(props) {
    super(props);
    this.state = { searchInput: '' };

    this.triggerSearch = this.triggerSearch.bind(this);
    this.triggerChange = this.triggerChange.bind(this);
    this.getInputValue = this.getInputValue.bind(this);
  }

  triggerChange(e, value, onChangeFunc) {
    if (onChangeFunc) {
      onChangeFunc(e);
    }
    if (value === undefined) {
      this.setState({ searchInput: e.target.value });
    }
  }

  triggerSearch(e) {
    const { onSearch, value } = this.props;
    const input = this.getInputValue(value, this.state);
    onSearch(e, input);
  }

  render() {
    const { placeholder, value, onChange } = this.props;
    return (
      <Spacer direction='horizontal' size='small'>
        <Input
          value={this.getInputValue(value, this.state)}
          onChange={(e) => this.triggerChange(e, value, onChange)}
          onPressEnter={this.triggerSearch}
          placeholder={placeholder}
          prefix={<i className='fas fa-search' style={{ fontStyle: 'normal' }} />}
          data-test-id='search-box'
          className={css(styles.searchBox)}
        />
        <span data-test-id='search-button'>
          <Button onClick={this.triggerSearch} data-test-id='search-button'>
            <FormattedMessage
              defaultMessage='Search'
              description='String for the search button to search objects in MLflow'
            />
          </Button>
        </span>
      </Spacer>
    );
  }
}

const styles = {
  searchBox: {
    height: '40px',
    padding: 0,
    borderRadius: 4,
    boxSizing: 'border-box',
    '.ant-input-prefix': {
      marginLeft: '16px',
      marginBottom: '2px',
      marginRight: '12px',
      left: 0,
    },
  },
};

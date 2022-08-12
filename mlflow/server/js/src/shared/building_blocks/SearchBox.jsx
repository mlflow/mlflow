import React from 'react';
import PropTypes from 'prop-types';
import { Spacer } from '../../shared/building_blocks/Spacer';
import { Input, Button } from '@databricks/design-system';
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

  componentDidUpdate(prevProps) {
    const valueChangedExternally = prevProps.value !== this.props.value;

    if (valueChangedExternally) {
      this.setState({ searchInput: this.props.value });
    }
  }

  constructor(props) {
    super(props);
    this.state = { searchInput: props.value };

    this.triggerSearch = this.triggerSearch.bind(this);
    this.triggerChange = this.triggerChange.bind(this);
  }

  triggerChange(e) {
    this.setState({ searchInput: e.target.value });
  }

  triggerSearch(e) {
    this.props.onSearch(e, this.state.searchInput);
  }

  render() {
    const { placeholder } = this.props;
    return (
      <Spacer direction='horizontal' size='small'>
        <Input
          value={this.state.searchInput}
          onChange={this.triggerChange}
          prefix={<i className='fas fa-search' style={{ fontStyle: 'normal' }} />}
          onPressEnter={this.triggerSearch}
          onBlur={this.props.onChange}
          placeholder={placeholder}
          data-test-id='search-box'
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

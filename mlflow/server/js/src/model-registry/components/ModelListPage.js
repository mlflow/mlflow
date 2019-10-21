import React from 'react';
import { ModelListView } from './ModelListView';
import { connect } from 'react-redux';
import { listRegisteredModelsApi } from '../actions';
import PropTypes from 'prop-types';
import { getUUID } from '../../Actions';
import RequestStateWrapper from '../../components/RequestStateWrapper';

class ModelListPage extends React.Component {
  static propTypes = {
    models: PropTypes.arrayOf(Object),
    listRegisteredModelsApi: PropTypes.func.isRequired,
  };

  listRegisteredModelsApiId = getUUID();

  componentDidMount() {
    this.props.listRegisteredModelsApi(this.listRegisteredModelsApiId);
  }

  render() {
    const { models } = this.props;
    return (
        <div className='App-content'>
          <RequestStateWrapper requestIds={[this.listRegisteredModelsApiId]}>
            <ModelListView models={models}/>
          </RequestStateWrapper>
        </div>
    );
  }
}

const mapStateToProps = (state) => {
  const models = Object.values(state.entities.modelByName);
  return { models };
};

const mapDispatchToProps = { listRegisteredModelsApi };

export default connect(mapStateToProps, mapDispatchToProps)(ModelListPage);

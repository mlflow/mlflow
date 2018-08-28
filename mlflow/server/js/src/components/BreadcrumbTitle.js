import React, { Component } from "react";
import PropTypes from "prop-types";
import { Experiment } from "../sdk/MlflowMessages";
import { Link } from 'react-router-dom';
import Routes from "../Routes";
import Utils from '../utils/Utils';
import DropdownMenuView from './DropdownMenuView';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';

/**
 * A title component that creates a <h1> with breadcrumbs pointing to an experiment and optionally
 * a run or a run comparison page.
 */

const DROPDOWN_MENU = 'dropdownMenu';


class BreadcrumbTitle extends Component {
  constructor(props) {
    super(props);
    this.onSetRunName = this.onSetRunName.bind(this);
    this.onSetTag = this.props.onSetTag.bind(this);
    // TODO do we need this here and below?
    this.state.showMenu = true;
  }
  // title={Utils.getRunDisplayName(this.props.tags, run.getRunUuid())}
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String), // Optional because not all pages are nested under runs
    // TODO this might need to be an array for multiple runs?
    tags: PropTypes.object,
    onSetTag: PropTypes.func,
  };

  state = {
      showMenu: true,
      above: false,
      dropdownHeight: 0,
  };

  onSetRunName(event) {
    event.preventDefault();
    if (event.target.value) {
      this.props.onSetTag(Utils.getRunTagName(), event.target.value);
    }
  }

  /**
   * Hide the dropdown menu for a table item.
   **/
  hideMenu() {
    this.setState({
      showMenu: true,
      above: false,
      dropdownHeight: 0,
    });
  }

  renameRun() {
    console.log("Hi! In renameRun() in BreadCrumbTitle.js");
  }

  getMenuItems() {
    // Table specific menu options
    const menuItems = [];

    const renameOnClick = this.renameRun.bind(this);
    menuItems.push(
      <a
        key='rename-item'
        data-name='Rename'
        className='sidebar-dropdown-link'
        onClick={renameOnClick}
      >Rename
      </a>
    );
    return menuItems;
  }

  renderDropdown() {
    return (<DropdownMenuView
            ref={DROPDOWN_MENU}
            getItems={this.getMenuItems}
            outsideClickHandler={this.hideMenu}
            ignoreClickClasses={['sidebar-dropdown']}
          />)
  }

  render() {
    const {experiment, runUuids, title} = this.props;
    const experimentId = experiment.getExperimentId();
    const experimentLink = (
      <Link to={Routes.getExperimentPageRoute(experimentId)}>
        {experiment.getName()}
      </Link>
    );
    let runsLink = null;
    if (runUuids) {
      runsLink = (runUuids.length === 1 ?
        <div>
          <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])} key="link">
            {Utils.getRunDisplayName(this.props.tags, runUuids[0])}
          </Link>
          <button>Hi from sid</button>
        </div>
        :
        <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)} key="link">
          Comparing {runUuids.length} Runs
        </Link>
      );
    }
    const chevron = <i className="fas fa-chevron-right breadcrumb-chevron" key="chevron"/>;
    return (
      <h1>
        {experimentLink}
        {chevron}
        { runsLink ? [runsLink, chevron] : [] }
        {title}
      </h1>
    );
  }
}


const mapStateToProps = (state, ownProps) => {
  const { experimentId, runUuids, onSetTag } = ownProps;
  const experiment = getExperiment(experimentId, state);
  // TODO handle array
  debugger;
  const params = getParams(runUuids[0], state);
  const tags = getRunTags(runUuids[0], state);
  const run = getRunInfo(runUuids[0], state);
  return { run, experiment, params, tags };
};


export default connect(mapStateToProps)(BreadcrumbTitle);

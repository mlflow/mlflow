import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactNbView from './ShowArtifactNbView';
import { Cell } from "@nteract/presentational-components";
import { emptyNotebook } from "@nteract/commutable";

test('ShowArtifactNbView notebook sample', () => {
  const wrapper = shallow(<ShowArtifactNbView
    path="fakepath"
    runUuid="fakerunuuid"
  />);
  wrapper.setState({immutableNotebook: emptyNotebook, loading: false, error: undefined})
  console.log(wrapper.text())
});


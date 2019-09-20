import React, { Component } from 'react';
import { shallow } from 'enzyme';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';


test('ShowArtifactHtmlView - simple website', (done) => {
  const simpleHtml = '<html><body><div id="html-test">Test Page</div></body></html>';
  const fetchMock = jest.fn();
  fetchMock.mockReturnValue(new Promise((resolve) => {
    resolve({
      blob: () => Promise.resolve(new Blob([simpleHtml])),
    });
  }));
  global.fetch = fetchMock;
  const wrapper = shallow(<ShowArtifactHtmlView path="fakepath" runUuid="fakerunuuid"/>);
  wrapper.instance().getBlobURL = () => {
    return simpleHtml;
  };
  wrapper.setState({html: simpleHtml, loading: false, error: undefined}, () => {
    wrapper.render();
    const iframe = wrapper.find('.html-iframe');
    const props = iframe.props();
    const nestedVal = iframe.find('.sid-test');
    // console.log(nestedVal.props());
    // console.log(props);
    return done();
  });
});

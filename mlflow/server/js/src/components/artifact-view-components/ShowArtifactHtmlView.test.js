import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';


test('ShowArtifactHtmlView - simple website', (done) => {
  const simpleHtml = '<html><body><div id="html-test">Test Page</div></body></html>';
  // Mock out fetch handler to silence react error from trying to fetch nonexistent file
  // The FileReader interface seems to reject the Blob we define below while reading it,
  // yielding an event of the form {isTrusted: false} instead of actual file contents,
  // so we manually set the component's HTML below. TODO(Sid): find a way to fix this
  // later on.
  const fetchMock = jest.fn();
  fetchMock.mockReturnValue(new Promise((resolve) => {
    resolve({
      blob: () => Promise.resolve(new Blob([simpleHtml])),
    });
  }));
  global.fetch = fetchMock;
  const wrapper = shallow(<ShowArtifactHtmlView path="fakepath" runUuid="fakerunuuid"/>);
  // Mock out our blob-to-url helper to return HTML directly as our URL, since unfortunately
  // we can't mock out URL.createObjectURL as per https://github.com/jsdom/jsdom/issues/1721
  wrapper.instance().getBlobURL = () => {
    return simpleHtml;
  };
  wrapper.setState({html: simpleHtml, loading: false, error: undefined}, () => {
    wrapper.render();
    const iframe = wrapper.find('.html-iframe');
    const props = iframe.props();
    expect(props.src).toEqual(simpleHtml);
    done();
  });
});

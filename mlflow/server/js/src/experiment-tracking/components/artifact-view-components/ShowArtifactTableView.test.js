import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactTableView from './ShowArtifactTableView';
import Papa from 'papaparse';

describe('ShowArtifactTableView', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakePath.csv',
      runUuid: 'fakeUuid',
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactTableView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactTableView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('render raw file text if parsing invalid CSV', (done) => {
    const fileContents = 'abcd\n&&&&&';
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(fileContents);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mount(<ShowArtifactTableView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.ShowArtifactPage').length).toBe(1);
      expect(wrapper.find('.text-area-border-box').length).toBe(1);
      expect(wrapper.find('.text-area-border-box').text()).toBe(fileContents);
      done();
    });
  });

  test('only show the first 500 rows when the number of rows is larger than 500', (done) => {
    const data = Array(600).fill({ a: 0, b: 1 });
    const fileContents = Papa.unparse(data);

    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(fileContents);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mount(<ShowArtifactTableView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('tbody').find('tr').length).toBe(500);
      done();
    });
  });
});

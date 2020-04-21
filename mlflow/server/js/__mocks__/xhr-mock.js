const xhrMockClass = () => ({
  open: jest.fn(),
  send: jest.fn(),
  status: 404
})

const xhr = window.XMLHttpRequest

export function setup_mock() {
  window.XMLHttpRequest = jest.fn().mockImplementation(xhrMockClass);
}

export function teardown_mock() {
  window.XMLHttpRequest = xhr;
}
  
window.privateVcsRegex = "some_regex";
window.privateVcsRepo = "repo_url";
window.privateVcsCommit = "commit_url";

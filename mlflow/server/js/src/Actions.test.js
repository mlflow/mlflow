import { ErrorWrapper, wrapDeferred } from './Actions';

test("ErrorWrapper.getErrorCode does not fail on JSON decoding problems", () => {
  new ErrorWrapper({ responseText: 'a{waefaw' });
});

test('wrapDeferred retries on 429s', () => {
  // mock Ajax
  const mockAjax = ({data, success, error}) => {
    return 0;
  };
  wrapDeferred(mockAjax, {});
});
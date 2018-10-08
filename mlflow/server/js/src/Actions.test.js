import { ErrorWrapper } from './Actions';

test("ErrorWrapper.getErrorCode does not fail on JSON decoding problems", () => {
  new ErrorWrapper({ responseText: 'a{waefaw' });
});

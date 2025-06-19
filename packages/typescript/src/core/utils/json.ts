import JSONBigInt from 'json-bigint';

// Configure json-bigint to handle large integers
const JSONBig = JSONBigInt({
  useNativeBigInt: true,
  alwaysParseAsBig: false,
  storeAsString: false
});

export { JSONBig };
import { gql } from '@apollo/client';

const GET_ENDPOINT_DETAILED_QUERY = gql`
    query testQuery {test(inputString: "abc") { output }}
`
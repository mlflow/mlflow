from copy import deepcopy
import urllib

from mlflow.utils import rest_utils

"""
This module provides utilities for performing Azure Blob Storage operations without requiring 
the heavyweight azure-storage-blob library dependency
"""

_PUT_BLOCK_HEADERS = {
    "x-ms-blob-type": "BlockBlob",
}


def put_block(sas_url, block_id, data, headers):
    """
    Performs an Azure `Put Block` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/put-block)

    :param sas_url: A shared access signature URL referring to the Azure Block Blob
                    to which the specified data should be staged.
    :param block_id: A base64-encoded string identifying the block.
    :param data: Data to include in the Put Block request body.
    :param headers: Additional headers to include in the Put Block request body
                    (the `x-ms-blob-type` header is always included automatically).
    """
    # Copy the headers to avoid mutating the input `headers` dictionary
    headers = deepcopy(headers)
    headers.update(_PUT_BLOCK_HEADERS)

    request_url = _append_query_parameters(
        sas_url,
        {
            "comp": "block",
            "blockid": block_id,
        }
    )

    with rest_utils.cloud_storage_http_request('put', request_url, data, headers=headers) as response:
        response.raise_for_status()


def put_block_list(sas_url, block_list, headers):
    """
    Performs an Azure `Put Block List` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/put-block-list)

    :param sas_url: A shared access signature URL referring to the Azure Block Blob
                    to which the specified data should be staged.
    :param block_list: A list of uncommitted base64-encoded string block IDs to commit. For
                       more information, see
                       https://docs.microsoft.com/en-us/rest/api/storageservices/put-block-list.
    :param headers: Headers to include in the Put Block request body.
    """
    request_url = _append_query_parameters(
        sas_url,
        {
            "comp": "blocklist",
        }
    )
    data = _build_block_list_xml(block_list)

    with rest_utils.cloud_storage_http_request('put', request_url, data, headers=headers) as response:
        response.raise_for_status()


def _append_query_parameters(url, parameters):
    parsed_url = urllib.parse.urlparse(url)
    query_dict = dict(urllib.parse.parse_qsl(parsed_url.query))
    query_dict.update(parameters)
    new_query = urllib.parse.urlencode(query_dict)
    new_url_components = parsed_url._replace(query=new_query)
    new_url = urllib.parse.urlunparse(new_url_components)
    return new_url


def _build_block_list_xml(block_list):
    xml = '<?xml version="1.0" encoding="utf-8"?>\n<BlockList>\n'
    for block_id in block_list:
        # Because block IDs are base64-encoded and base64 strings do not contain
        # XML special characters, we can safely insert the block ID directly into
        # the XML document
        xml += "<Uncommitted>{}</Uncommitted>\n".format(block_id)
    xml += "</BlockList>"
    return xml

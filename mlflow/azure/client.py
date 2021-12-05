"""
This module provides utilities for performing Azure Blob Storage operations without requiring
the heavyweight azure-storage-blob library dependency
"""
from copy import deepcopy
import urllib
import logging

from mlflow.utils import rest_utils

_logger = logging.getLogger(__name__)
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
    request_url = _append_query_parameters(sas_url, {"comp": "block", "blockid": block_id})

    request_headers = deepcopy(_PUT_BLOCK_HEADERS)
    for name, value in headers.items():
        if _is_valid_put_block_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for Put Block operation", name)

    with rest_utils.cloud_storage_http_request(
        "put", request_url, data=data, headers=request_headers
    ) as response:
        rest_utils.augmented_raise_for_status(response)


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
    request_url = _append_query_parameters(sas_url, {"comp": "blocklist"})
    data = _build_block_list_xml(block_list)

    request_headers = {}
    for name, value in headers.items():
        if _is_valid_put_block_list_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for Put Block List operation", name)

    with rest_utils.cloud_storage_http_request(
        "put", request_url, data=data, headers=request_headers
    ) as response:
        rest_utils.augmented_raise_for_status(response)


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


def _is_valid_put_block_list_header(header_name):
    """
    :return: True if the specified header name is a valid header for the Put Block List operation,
             False otherwise. For a list of valid headers, see https://docs.microsoft.com/en-us/
             rest/api/storageservices/put-block-list#request-headers and https://docs.microsoft.com/
             en-us/rest/api/storageservices/
             specifying-conditional-headers-for-blob-service-operations#Subheading1.
    """
    return header_name.startswith("x-ms-meta-") or header_name in set(
        [
            "Authorization",
            "Date",
            "x-ms-date",
            "x-ms-version",
            "Content-Length",
            "Content-MD5",
            "x-ms-content-crc64",
            "x-ms-blob-cache-control",
            "x-ms-blob-content-type",
            "x-ms-blob-content-encoding",
            "x-ms-blob-content-language",
            "x-ms-blob-content-md5",
            "x-ms-encryption-scope",
            "x-ms-tags",
            "x-ms-lease-id",
            "x-ms-client-request-id",
            "x-ms-blob-content-disposition",
            "x-ms-access-tier",
            "If-Modified-Since",
            "If-Unmodified-Since",
            "If-Match",
            "If-None-Match",
        ]
    )


def _is_valid_put_block_header(header_name):
    """
    :return: True if the specified header name is a valid header for the Put Block operation, False
             otherwise. For a list of valid headers, see
             https://docs.microsoft.com/en-us/rest/api/storageservices/put-block#request-headers and
             https://docs.microsoft.com/en-us/rest/api/storageservices/put-block#
             request-headers-customer-provided-encryption-keys.
    """
    return header_name in set(
        [
            "Authorization",
            "x-ms-date",
            "x-ms-version",
            "Content-Length",
            "Content-MD5",
            "x-ms-content-crc64",
            "x-ms-encryption-scope",
            "x-ms-lease-id",
            "x-ms-client-request-id",
            "x-ms-encryption-key",
            "x-ms-encryption-key-sha256",
            "x-ms-encryption-algorithm",
        ]
    )

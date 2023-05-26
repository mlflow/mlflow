"""
This module provides utilities for performing Azure Blob Storage operations without requiring
the heavyweight azure-storage-blob library dependency
"""
from copy import deepcopy
import urllib
import logging

from mlflow.utils import rest_utils
from mlflow.utils.file_utils import read_chunk

_logger = logging.getLogger(__name__)
_PUT_BLOCK_HEADERS = {
    "x-ms-blob-type": "BlockBlob",
}


def put_adls_file_creation(sas_url, headers):
    """
    Performs an ADLS Azure file create `Put` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/create)

    :param sas_url: A shared access signature URL referring to the Azure ADLS server
                    to which the file creation command should be issued.
    :param headers: Additional headers to include in the Put request body
    """
    request_url = _append_query_parameters(sas_url, {"resource": "file"})

    request_headers = {}
    for name, value in headers.items():
        if _is_valid_adls_put_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for ADLS Gen2 Put operation", name)

    with rest_utils.cloud_storage_http_request(
        "put", request_url, headers=request_headers
    ) as response:
        rest_utils.augmented_raise_for_status(response)


def patch_adls_file_upload(sas_url, local_file, start_byte, size, position, headers, is_single):
    """
    Performs an ADLS Azure file create `Patch` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/update)

    :param sas_url: A shared access signature URL referring to the Azure ADLS server
                    to which the file update command should be issued.
    :param local_file: The local file to upload
    :param start_byte: The starting byte of the local file to upload
    :param size: The number of bytes to upload
    :param position: Positional offset of the data in the Patch request
    :param headers: Additional headers to include in the Patch request body
    :param is_single: Whether this is the only patch operation for this file
    """
    new_params = {"action": "append", "position": str(position)}
    if is_single:
        new_params["flush"] = "true"
    request_url = _append_query_parameters(sas_url, new_params)

    request_headers = {}
    for name, value in headers.items():
        if _is_valid_adls_patch_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for ADLS Gen2 Patch operation", name)

    data = read_chunk(local_file, size, start_byte)
    with rest_utils.cloud_storage_http_request(
        "patch", request_url, data=data, headers=request_headers
    ) as response:
        rest_utils.augmented_raise_for_status(response)


def patch_adls_flush(sas_url, position, headers):
    """
    Performs an ADLS Azure file flush `Patch` operation
    (https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/update)

    :param sas_url: A shared access signature URL referring to the Azure ADLS server
                    to which the file update command should be issued.
    :param position: The final size of the file to flush.
    :param headers: Additional headers to include in the Patch request body.
    """
    request_url = _append_query_parameters(sas_url, {"action": "flush", "position": str(position)})

    request_headers = {}
    for name, value in headers.items():
        if _is_valid_adls_put_header(name):
            request_headers[name] = value
        else:
            _logger.debug("Removed unsupported '%s' header for ADLS Gen2 Patch operation", name)

    with rest_utils.cloud_storage_http_request(
        "patch", request_url, headers=request_headers
    ) as response:
        rest_utils.augmented_raise_for_status(response)


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
        xml += f"<Uncommitted>{block_id}</Uncommitted>\n"
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
    return header_name.startswith("x-ms-meta-") or header_name in {
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
    }


def _is_valid_put_block_header(header_name):
    """
    :return: True if the specified header name is a valid header for the Put Block operation, False
             otherwise. For a list of valid headers, see
             https://docs.microsoft.com/en-us/rest/api/storageservices/put-block#request-headers and
             https://docs.microsoft.com/en-us/rest/api/storageservices/put-block#
             request-headers-customer-provided-encryption-keys.
    """
    return header_name in {
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
    }


def _is_valid_adls_put_header(header_name):
    """
    :return: True if the specified header name is a valid header for the ADLS Put operation, False
             otherwise. For a list of valid headers, see
             https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/create
    """
    return header_name in {
        "Cache-Control",
        "Content-Encoding",
        "Content-Language",
        "Content-Disposition",
        "x-ms-cache-control",
        "x-ms-content-type",
        "x-ms-content-encoding",
        "x-ms-content-language",
        "x-ms-content-disposition",
        "x-ms-rename-source",
        "x-ms-lease-id",
        "x-ms-properties",
        "x-ms-permissions",
        "x-ms-umask",
        "x-ms-owner",
        "x-ms-group",
        "x-ms-acl",
        "x-ms-proposed-lease-id",
        "x-ms-expiry-option",
        "x-ms-expiry-time",
        "If-Match",
        "If-None-Match",
        "If-Modified-Since",
        "If-Unmodified-Since",
        "x-ms-source-if-match",
        "x-ms-source-if-none-match",
        "x-ms-source-if-modified-since",
        "x-ms-source-if-unmodified-since",
        "x-ms-encryption-key",
        "x-ms-encryption-key-sha256",
        "x-ms-encryption-algorithm",
        "x-ms-encryption-context",
        "x-ms-client-request-id",
        "x-ms-date",
        "x-ms-version",
    }


def _is_valid_adls_patch_header(header_name):
    """
    :return: True if the specified header name is a valid header for the ADLS Patch operation, False
             otherwise. For a list of valid headers, see
             https://docs.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/update
    """
    return header_name in {
        "Content-Length",
        "Content-MD5",
        "x-ms-lease-id",
        "x-ms-cache-control",
        "x-ms-content-type",
        "x-ms-content-disposition",
        "x-ms-content-encoding",
        "x-ms-content-language",
        "x-ms-content-md5",
        "x-ms-properties",
        "x-ms-owner",
        "x-ms-group",
        "x-ms-permissions",
        "x-ms-acl",
        "If-Match",
        "If-None-Match",
        "If-Modified-Since",
        "If-Unmodified-Since",
        "x-ms-encryption-key",
        "x-ms-encryption-key-sha256",
        "x-ms-encryption-algorithm",
        "x-ms-encryption-context",
        "x-ms-client-request-id",
        "x-ms-date",
        "x-ms-version",
    }

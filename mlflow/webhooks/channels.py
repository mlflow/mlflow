import logging
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import requests

from mlflow.entities.model_registry.webhook import (
    Webhook,
    WebhookEventTrigger,
)
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)

# Minimum time between webhook database refreshes to prevent overload
WEBHOOKS_MIN_REFRESH_INTERVAL = 5  # seconds


@dataclass
class RegisteredModelEvent:
    """
    Base class for all model registry events.
    """

    model_name: str
    model_version: str
    event_trigger: WebhookEventTrigger
    key: str
    value: str


class EventsChannel:
    """
    A utility class that manages message channels for internal communication.

    Attributes:
        _channel: Queue instance that serves as the message channel
        _max_size: Maximum number of messages the channel can hold
    """

    def __init__(self, max_size: Optional[int] = 0):
        """
        Initialize the channel and start background tasks to fetch webhooks
        and create listeners.
        """
        self._max_size = max_size
        self._channel = queue.Queue(maxsize=max_size)
        self._session = requests.Session()
        self._workers = []
        self._worker_count = os.cpu_count() or 1
        self._running = True
        _logger.info(
            f"Initializing EventsChannel with max_size={max_size}, workers={self._worker_count}"
        )

        # Webhook synchronization and caching
        self._mlflow_client = MlflowClient()
        self.webhooks = []
        self._webhook_lock = threading.RLock()  # Reentrant lock for thread safety
        self._last_webhook_update = 0  # Timestamp of last update
        self._webhook_update_in_progress = False  # Flag to prevent concurrent updates

        # Fetch webhooks from the data store initially
        self._update_webhooks(force=True)

        # Create a thread pool for sending webhook requests
        max_workers = min(10, (os.cpu_count() or 1) * 2)
        self._webhook_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="Webhook-Sender"
        )

        self._initialize_consumers()

    def _update_webhooks(self, force=False):
        """
        Fetches the latest webhooks from the data store with thread synchronization.

        Args:
            force: If True, forces an update regardless of the time since last update

        Returns:
            bool: True if webhooks were updated, False if skipped
        """
        current_time = time.time()

        # Use a lock to ensure thread safety when checking conditions
        with self._webhook_lock:
            # Skip if another thread is already updating
            if self._webhook_update_in_progress:
                return False

            # Skip if updated recently (unless forced)
            time_since_update = current_time - self._last_webhook_update
            if not force and time_since_update < WEBHOOKS_MIN_REFRESH_INTERVAL:
                return False

            # Mark as updating to prevent other threads from initiating updates
            self._webhook_update_in_progress = True

        try:
            # Fetch webhooks outside the lock to minimize lock contention
            _logger.debug("Fetching updated webhooks from database")
            new_webhooks = self._mlflow_client.search_webhooks()

            # Update webhooks with lock
            with self._webhook_lock:
                self.webhooks = new_webhooks
                self._last_webhook_update = current_time
                _logger.debug(f"Updated webhooks: {len(self.webhooks)} found")
                return True
        except Exception as e:
            _logger.error(f"Error updating webhooks: {e}")
            return False
        finally:
            # Always clear the in-progress flag when done
            with self._webhook_lock:
                self._webhook_update_in_progress = False

    def _get_current_webhooks(self):
        """
        Gets the current webhook list, ensuring it's up to date before processing.

        Returns:
            list: The current list of webhooks
        """
        # Try to update webhooks if needed, but don't force it if updated recently
        self._update_webhooks(force=False)

        # Return the current webhooks (thread-safe with lock)
        with self._webhook_lock:
            return self.webhooks.copy()  # Return a copy to avoid thread issues

    def _initialize_consumers(self):
        """Start listeners in the background to listen for incoming events."""
        for i in range(self._worker_count):
            worker = threading.Thread(
                target=self._listen_for_webhook_events,
                daemon=True,
                name=f"EventsChannel-Worker-{i}",
            )
            self._workers.append(worker)
            worker.start()
            _logger.debug(f"Started worker thread {worker.name}")

    def _listen_for_webhook_events(self):
        """
        Main worker loop that continuously processes messages from the queue.
        Blocks until a message is available, then processes it.
        """
        thread_name = threading.current_thread().name
        _logger.debug(f"{thread_name} started and waiting for messages")

        while self._running:
            try:
                message = self._channel.get(block=True)

                try:
                    self._handle_event(message)
                except Exception as e:
                    _logger.error(f"{thread_name} encountered error processing message: {e}")
                finally:
                    self._channel.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                _logger.error(f"{thread_name} encountered error: {e}")

    def _handle_event(self, event: RegisteredModelEvent):
        """
        Process an event by filtering webhooks and sending requests to matching ones.

        Args:
            event: The event to be processed.
        """
        _logger.debug(f"Processing event of type {event.event_trigger}")

        # Filter webhooks based on event type
        matching_webhooks = []
        for webhook in self._get_current_webhooks():
            if (
                event.event_trigger.value == webhook.event_trigger
                and event.key == webhook.key
                and event.value == webhook.value
            ):
                matching_webhooks.append(webhook)

        if not matching_webhooks:
            _logger.debug(f"No webhooks matched event type {event.event_trigger}")
            return

        _logger.debug(
            f"Found {len(matching_webhooks)} matching webhooks for event {event.event_trigger}"
        )

        # Submit webhook requests to thread pool for parallel processing
        futures = []
        for webhook in matching_webhooks:
            future = self._webhook_executor.submit(self._send_webhook, webhook, event)
            futures.append(future)

    def _send_webhook(self, webhook: Webhook, event: RegisteredModelEvent):
        """
        Send a webhook request to the specified URL.

        Args:
            webhook: The webhook configuration
            event: The event that triggered the webhook
        """
        thread_name = threading.current_thread().name
        _logger.debug(f"[{thread_name}] Sending webhook request to {webhook.url}")
        try:
            extra_payload = {
                "webhook_name": webhook.name,
                "event_trigger": event.event_trigger.value,
            }
            payload = webhook.payload | extra_payload
            try:
                response = self._session.post(
                    webhook.url,
                    headers=webhook.headers,
                    json=payload,
                    timeout=5.0,
                    verify=False,
                )
                _logger.debug(
                    f"[{thread_name}] Request completed with status: {response.status_code}"
                )
                _logger.debug(f"[{thread_name}] Response body: {response.text[:200]}")
            except requests.exceptions.Timeout as timeout_err:
                _logger.warning(
                    f"""[{thread_name}] Request for webhook {webhook.name} to
                    '{webhook.url}' timed out: {timeout_err}"""
                )
                return
            except requests.exceptions.ConnectionError as conn_err:
                _logger.warning(
                    f"""[{thread_name}] Connection error for webhook {webhook.name}
                    to '{webhook.url}': {conn_err}"""
                )
                if hasattr(conn_err, "args") and conn_err.args:
                    _logger.warning(f"[{thread_name}] Connection error details: {conn_err.args[0]}")
                return
            except Exception as req_error:
                _logger.warning(
                    f"""[{thread_name}] Request failed for webhook {webhook.name} to
                    '{webhook.url}' with: {type(req_error).__name__}: {req_error}"""
                )

                _logger.warning(f"[{thread_name}] Traceback: {traceback.format_exc()}")
                return

        except Exception as e:
            _logger.warning(
                f"""[{thread_name}] Error in webhook {webhook.name} to
                '{webhook.url}' processing: {e}"""
            )

            _logger.warning(f"[{thread_name}] Traceback: {traceback.format_exc()}")

    def send(self, message: RegisteredModelEvent) -> queue.Queue:
        """
        Sends a message to the channel.
        This is a non blocking operation: The message will be dropped if the channel is full,
        and a warning will be logged.

        Args:
            message: The message to be sent. Must be one of the supported event types.
        """
        try:
            self._channel.put(message, block=False)
        except queue.Full:
            _logger.warning("Channel is full. Dropping message.")

    def shutdown(self, wait: bool = True):
        """
        Shutdown the worker threads.

        Args:
            wait: If True, wait for all tasks in the queue to be processed before shutdown
        """
        if wait:
            # Wait for all tasks to be processed
            self._channel.join()

        # Signal workers to stop
        self._running = False

        # Wait for all workers to finish
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=1.0)

        # Close the session
        self._session.close()

        # Shutdown the executor
        self._webhook_executor.shutdown(wait=wait)

        _logger.info("EventsChannel shutdown complete")

    @property
    def max_size(self) -> int:
        """
        Returns maximum size of the channel.

        Returns:
            int: Maximum queue size
        """
        return self._max_size

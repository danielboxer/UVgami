"""Cancellation for the long solves.

A thread cannot be killed, so the caller passes a cancelled callable down and
the passes check it where giving up is safe. Every check raises the same
exception, so a cancel reads apart from a failure."""


class Cancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise Cancelled

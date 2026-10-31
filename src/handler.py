import bpy

from .logger import logger
from .manager import manager
from .utils.ui import popup


def _log_and_show(error, heading):
    error_lines = logger.log_traceback(heading)
    logger.update_time()
    logger.get_latest().status = "Error"
    logger.write_latest()
    popup(error_lines, heading + str(error), "ERROR")


def handle_start_error(error, start_objects):
    _log_and_show(error, "Error starting unwrap: ")
    if manager.is_active:
        return
    for obj in set(bpy.data.objects).difference(start_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    manager.end_session()


def handle_dispatch_error(error):
    _log_and_show(error, "Error during unwrap: ")
    manager.end_session()

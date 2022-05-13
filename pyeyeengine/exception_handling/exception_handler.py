# import sys
#
# import os.path
# import traceback
#
# def handle_exception(exc_type, exc_value, exc_traceback):
#   """ handle all exceptions """
#
#   ## KeyboardInterrupt is a special case.
#   ## We don't raise the error dialog when it occurs.
#   if issubclass(exc_type, KeyboardInterrupt):
#     sys.exit(0)
#     return
#
#   filename, line, dummy, dummy = traceback.extract_tb( exc_traceback ).pop()
#   filename = os.path.basename( filename )
#   error    = "%s: %s" % ( exc_type.__name__, exc_value )
#
#   print("Closed due to an error in file '{}' line '{}'. Error: {}".format(filename, line, error))
#   print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
#   sys.exit(1)

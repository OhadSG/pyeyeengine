import ctypes as c

gi_dll  = c.cdll.LoadLibrary('EyeclickGI.dll')
gi_dll.PushMesssageSimple(c.c_float(0),c.c_float(0),c.c_float(0),c.c_float(0),c.c_float(0))
# works !!!!!
import usb
import ctypes

device = usb.core.find(idVendor=0x2bc5)
device.set_configuration()
endpt = device[0][(0,0)][1]
#endpt = 0x82 # print(device)

device.write(endpt,  bytes([71,77,2,0,83,0,204,204,24,80,0,80]))
# hi=5



import usb.core
import usb.util

# find our device
dev = usb.core.find(idVendor=0x2bc5)

# was it found?
if dev is None:
    raise ValueError('Device not found')

# set the active configuration. With no arguments, the first
# configuration will be the active one
dev.set_configuration()

# get an endpoint instance
cfg = dev.get_active_configuration()
intf = cfg[(0,0)]

ep = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    custom_match = \
    lambda e: \
        usb.util.endpoint_direction(e.bEndpointAddress) == \
        usb.util.ENDPOINT_OUT)

assert ep is not None
dev.write(ep, bytes([71,77,2]))
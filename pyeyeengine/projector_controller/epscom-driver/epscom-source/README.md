##########


USAGE:

#1
epscom-cmd <string "cmd">

Takes exactly one argument

EXAMPLE: $epscom-cmd "PWR ON"

#2
epscom-dbg <string "cmd"> <int delay>
Runs command with debug information
Takes exactly two arguments
To avoid delay use 0 as second argument 

EXAMPLE: $epscom-dbg "LAMP?" 5

NOTE UGLY WORKAROUND: Clearing buffer by proper usb request would have taken many moons of research, so I have made an ugly workaround as such: once the relevant information received from the projector the driver keeps reading the device buffer and disregard additional data until device buffer is empty.
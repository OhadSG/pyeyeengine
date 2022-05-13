# > make
# will create both the command executable and the debug executable
all: cmd dbg

# > make cmd
# will create the command executable
cmd:
	gcc user-command.c -lusb-1.0 -o epscom-cmd

# > make dbd
# will create the debug executable
dbg:
	gcc user-command.c -lusb-1.0 -DEPSCOM_DEBUG=1 -o epscom-dbg

# > make clean
# will remove old executables
clean:
	rm epscom-cmd epscom-dbg
# Compiler
CC = gcc
export MACOSX_DEPLOYMENT_TARGET=13.0

# Paths
TCLTK_PREFIX := /opt/homebrew/opt/tcl-tk@8
X11_PREFIX := /opt/X11

# Compiler flags
CFLAGS = -g -Wall -Wno-pointer-sign -Wno-deprecated-declarations \
         -I$(TCLTK_PREFIX)/include \
         -I$(X11_PREFIX)/include

# Linker flags and libraries (link to Homebrew dylibs directly)
LIBS = \
  $(TCLTK_PREFIX)/lib/libtcl8.6.dylib \
  $(TCLTK_PREFIX)/lib/libtk8.6.dylib \
  -L$(X11_PREFIX)/lib -lX11 -lm

# rpath so dylibs are found at runtime
LDFLAGS = -Wl,-rpath,$(TCLTK_PREFIX)/lib

# Executable and objects
EXE = myWish
OBJS = tkAppInit.o PacWarGuts.o PacWarTk.o
SIM_AUX = PacWar.tcl

# Default target
all: $(EXE)

# Link executable
$(EXE): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(EXE)

# Compile .c files
.c.o:
	$(CC) -c $(CFLAGS) $<

# Clean build artifacts
clean:
	rm -rf $(EXE) $(OBJS)

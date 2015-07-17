# Makefile, lab7, Tim Goodwin
#
#


CC = gcc
CXX = g++

INCLUDES =
CFLAGS = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)

LDFLAGS =
LDLIBS =

.PHONY: default
default: http-server

http-server: http-server.o 

http-server.o: http-server.c

.PHONY: clean
clean: 
	rm -f *.o *~ a.out core http-server


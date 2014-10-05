# EE569 Homework Assignment #1
# Date: Sept 21, 2014
# Name: Rishabh Battulwar
# ID: 4438-1435-20
# email: battulwa@usc.edu
#
# Compiled on CYGWIN with g++

CXX = g++
CXXFLAGS = -Wall -g -O -Wno-write-strings
LDFLAGS = -lm
INCLUDES = -I./include

Main : ./src/Main.cpp
	if [ ! -d ./build ];	then	mkdir build;	fi
	if [ ! -d ./images ];	then	mkdir images;	fi
	${CXX} ${CXXFLAGS} ${INCLUDES} -c ./src/Main.cpp && mv Main.o ./build/
	${CXX} ${CXXFLAGS} ./build/Main.o -o ./build/Main


rebuild:
	rm -rf ./build/Main ./build/Main.o
	if [ ! -d ./build ];	then	mkdir build;	fi
	if [ ! -d ./images ];	then	mkdir images;	fi
	${CXX} ${CXXFLAGS} ${INCLUDES} -c ./src/Main.cpp && mv Main.o ./build/
	${CXX} ${CXXFLAGS} ./build/Main.o -o ./build/Main
	
.PHONY : clean

clean :
	rm -rf ./build/Main ./build/Main.o



#-g - turn on debugging (so GDB gives more friendly output)
#-Wall - turns on most warnings
#-O or -O2 - turn on optimizations
#-o <name> - name of the output file
#-c - output an object file (.o)
#-I<include path> - specify an include directory
#-L<library path> - specify a lib directory
#-l<library> - link with library lib<library>.a
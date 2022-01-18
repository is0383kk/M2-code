# Makefile
ALL: Multi_dist.so Multi_dist.o
Multi_dist.so: Multi_dist.o
	gcc -shared -o Multi_dist.so Multi_dist.o
Multi_dist.o:multiWrapper.c
	gcc -fPIC -Wall -c -o Multi_dist.o  multiWrapper.c

 all: 
	g++ main.cpp `pkg-config --cflags --libs opencv` -I/usr/include/SDL -lSDL -lSDL_ttf -std=c++11  

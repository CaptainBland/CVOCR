all: 
		g++ main.cpp `pkg-config --cflags --libs opencv` -I/usr/include/SDL -lSDL -lSDL_ttf -std=c++11
	
nntest: 
		g++ NeuralNet.cpp `pkg-config --cflags --libs opencv` -I/usr/include/SDL -lSDL -lSDL_ttf -std=c++11 -o nn.out -DENABLE_LOGGING

clean:
		rm *.o*

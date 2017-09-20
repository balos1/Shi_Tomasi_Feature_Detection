CC=g++
CUCC=nvcc
CCFLAGS=-Wall
CUFLAGS=-std=c++11
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lm
ENVFLAGS=-DNBLOCKS=16
SOURCES=$(wildcard *.c)
CUSOURCES=$(wildcard *.cu)
OBJECTS=$(SOURCES:.c=.o)
CUOBJECTS=$(CUSOURCES:.cu=.o)
TARGET=stfd

all: debug

debug: CCFLAGS += -DDEBUG -g
debug: CUFLAGS += -DDEBUG -g
debug: $(TARGET)

release: CCFLAGS += -O2
release: CUFLAGS += -O2
release: $(TARGET)

benchmode: CCFLAGS += -O2 -DBENCHMARKMODE
benchmode: CUFLAGS += -O2 -DBENCHMARKMODE
benchmode: $(TARGET)

$(TARGET): $(OBJECTS) $(CUOBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CCFLAGS) $(ENVFLAGS) -c -I/usr/local/cuda/include $<

%.o: %.c
	$(CC) $(CCFLAGS) $(ENVFLAGS) -c -I/usr/local/cuda/include $<

%.o: %.cu %.h
	$(CUCC) $(CUFLAGS) $(ENVFLAGS) -c $<

%.o: %.cu
	$(CUCC) $(CUFLAGS) $(ENVFLAGS) -c $<

clean:
	rm -f *.pgm *.o $(TARGET)
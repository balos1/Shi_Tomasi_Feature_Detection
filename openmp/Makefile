CC=gcc
CCFLAGS=-Wall -fopenmp
LDFLAGS=-fopenmp -lm
SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)
TARGET=stfd

all: debug

debug: CCFLAGS += -DDEBUG -g
debug: $(TARGET)

release: CCFLAGS += -O2
release: $(TARGET)

benchmode: CCFLAGS += -O2 -DBENCHMARKMODE
benchmode: $(TARGET)

$(TARGET): $(OBJECTS) $(CXXOBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c %.h
	$(CC) $(CCFLAGS) -c $<

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

clean:
	rm -f *.pgm *.o $(TARGET)
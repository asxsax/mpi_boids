MPICXX = mpicxx
CXXFLAGS = -O2 -Wall
TARGET = bird_flock
SRCS = bird_flock.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)


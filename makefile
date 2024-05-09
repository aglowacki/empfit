# options
CC=clang
CFLAGS=-W -Wall -g -lstdc++ -D__Unix__ -march=native -O3  -std=c++1z -arch arm64 -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk -fPIC -DDARWIN -I/Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk/usr/include
TARGET=bin/test

# globs
SRCS := $(wildcard test/*.cpp)
HDRS := $(wildcard src/*.hpp)
OBJS := $(patsubst test/%.cpp,bin/%.o,$(SRCS))

# link it all together
$(TARGET): $(OBJS) $(HDRS) makefile
	@mkdir -p bin
	$(CC) $(CFLAGS) $(OBJS) -Isrc -I../XRF-Maps/src/support/eigen-git-mirror -o $(TARGET)

# compile an object based on source and headers
$(OBJS): $(SRCS) $(HDRS) makefile
	@mkdir -p bin
	$(CC) $(CFLAGS) -Isrc -I../XRF-Maps/src/support/eigen-git-mirror -c $< -o $@

# tidy up
clean:
	rm -f $(TARGET) $(OBJS)

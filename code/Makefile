CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP -fopenmp -O3 -march=native
LDFLAGS := -fopenmp
TARGET := solve_cvrptw

SRCS := \
	solve_cvrptw.cpp \
	lib/vrp.cpp \
	lib/route_utils.cpp \
	lib/cluster/clustering.cpp \
	lib/clark/clarke_wright.cpp \
	lib/optim/intra_route_optimization.cpp \
	lib/optim/inter_route_optimization.cpp

OBJS := $(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) r102.txt

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)
	rm -rf outputs

-include $(DEPS)

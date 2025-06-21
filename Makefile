CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
LDFLAGS =

TARGETS = el_farol_spatial process_binary_data

all: $(TARGETS)

el_farol_spatial: el_farol_spatial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

process_binary_data: process_binary_data.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) p_values.bin entropy_results.csv histogram_data.bin *.o 
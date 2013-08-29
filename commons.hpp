#ifndef COMMONS
#define COMMONS
#include "OpenCLUtilities/openCLUtilities.hpp"
typedef struct OpenCL {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
} OpenCL;


#endif

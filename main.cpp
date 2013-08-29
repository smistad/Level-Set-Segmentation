#include "SIPL/Core.hpp"
#include "SIPL/Visualization.hpp"
#include "commons.hpp"
#include "histogram-pyramids.hpp"
#include <iostream>
using namespace SIPL;
using namespace std;

#define MAX(a,b) (a > b ? a : b)
#define MIN(a,b) (a < b ? a : b)

#ifndef KERNELS_DIR
#define KERNELS_DIR ""
#endif
void updateLevelSetFunction(
        OpenCL &ocl,
        cl::Kernel &kernel,
        cl::Image3D &input,
        cl::Buffer &positions,
        int activeVoxels,
        cl::Image3D &phi_read,
        cl::Image3D &phi_write,
        float threshold,
        float epsilon,
        float alpha
        ) {

    kernel.setArg(0, input);
    kernel.setArg(1, positions);
    kernel.setArg(2, activeVoxels);
    kernel.setArg(3, phi_read);
    kernel.setArg(4, phi_write);
    kernel.setArg(5, threshold);
    kernel.setArg(6, epsilon);
    kernel.setArg(7, alpha);

    ocl.queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(activeVoxels),
            cl::NullRange
    );
}

void visualize(Volume<float> * input, Volume<float> * phi, float level, float window) {
    Volume<char> * seg = new Volume<char>(input->getSize());
    for(int i = 0; i < input->getTotalSize(); i++) {
        char value = 0;
        if(phi->get(i) < 0) {
            value = 1;
        }
        seg->set(i,value);
    }
    Visualization * v = new Visualization(input, seg);
    v->setLevel(input, level);
    v->setWindow(input, window);
    v->display();
}

Volume<float> * runLevelSet(
        OpenCL &ocl,
        Volume<float> * input,
        int3 seedPos,
        float seedRadius,
        int iterations,
        float threshold,
        float epsilon,
        float alpha
        ) {
    int3 size = input->getSize();
    cl::Image3D inputData = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_R, CL_FLOAT),
            input->getWidth(),
            input->getHeight(),
            input->getDepth(),
            0,0,
            (float *)input->getData()
    );

    cl::Image3D phi_1 = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_R, CL_FLOAT),
            input->getWidth(),
            input->getHeight(),
            input->getDepth()
    );
    cl::Image3D phi_2 = cl::Image3D(
        ocl.context,
        CL_MEM_READ_WRITE,
        cl::ImageFormat(CL_R, CL_FLOAT),
        input->getWidth(),
        input->getHeight(),
        input->getDepth()
    );

    cl::Image3D activeSet = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_R, CL_SIGNED_INT8),
            input->getWidth(),
            input->getHeight(),
            input->getDepth()
    );


    // Create seed
    char narrowBandDistance = 10;
    cl::Kernel createSeedKernel(ocl.program, "initializeLevelSetFunction");
    createSeedKernel.setArg(0, phi_1);
    createSeedKernel.setArg(1, seedPos.x);
    createSeedKernel.setArg(2, seedPos.y);
    createSeedKernel.setArg(3, seedPos.z);
    createSeedKernel.setArg(4, seedRadius);
    createSeedKernel.setArg(5, activeSet);
    createSeedKernel.setArg(6, narrowBandDistance);
    createSeedKernel.setArg(7, phi_2);
    ocl.queue.enqueueNDRangeKernel(
            createSeedKernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );

    HistogramPyramid3D hp(ocl);
    hp.create(activeSet, size.x, size.y, size.z);
    int activeVoxels = hp.getSum();
    std::cout << "Number of active voxels: " << activeVoxels << std::endl;
    cl::Buffer positions = hp.createPositionBuffer();

    cl::Kernel kernel(ocl.program, "updateLevelSetFunction");
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;

    for(int i = 0; i < iterations; i++) {
        if(i % 2 == 0) {
            updateLevelSetFunction(ocl, kernel, inputData, positions, activeVoxels, phi_1, phi_2, threshold, epsilon, alpha);
        } else {
            updateLevelSetFunction(ocl, kernel, inputData, positions, activeVoxels, phi_2, phi_1, threshold, epsilon, alpha);
        }
    }
    if(iterations % 2 != 0) {
        // Phi_2 was written to in the last iteration, copy this to the result
        ocl.queue.enqueueCopyImage(phi_2,phi_1,origin,origin,region);
    }

    Volume<float> * phi = new Volume<float>(input->getSize());
    float * data = (float *)phi->getData();
    ocl.queue.enqueueReadImage(
            phi_1,
            CL_TRUE,
            origin,
            region,
            0, 0,
            data
    );

    phi->setData(data);

    return phi;
}

int main(int argc, char ** argv) {

    if(argc < 11) {
        cout << endl;
        cout << "OpenCL Level Set Segmentation by Erik Smistad 2013" << endl;
        cout << "www.github.com/smistad/OpenCL-Level-Set-Segmentation/" << endl;
        cout << "======================================================" << endl;
        cout << "The speed function is defined as -alpha*(epsilon-(T-intensity))+(1-alpha)*curvature" << endl;
        cout << "Usage: " << argv[0] << " inputFile.mhd outputFile.mhd seedX seedY seedZ seedRadius iterations threshold epsilon alpha [level window]" << endl;
        cout << "If the level and window arguments are set, the segmentation result will be displayed as an overlay to the input volume " << endl;
        return -1;
    }

    // Create OpenCL context
    OpenCL ocl;
    ocl.context = createCLContext(CL_DEVICE_TYPE_GPU, VENDOR_ANY);
    VECTOR_CLASS<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.device = devices[0];
    ocl.queue = cl::CommandQueue(ocl.context, devices[0]);
    string filename = string(KERNELS_DIR) + string("kernels.cl");
    string buildOptions = "";
    if(ocl.device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") == 0)
        buildOptions = "-DNO_3D_WRITE";
    ocl.program = buildProgramFromSource(ocl.context, filename);

    // Load volume
    Volume<float> * input = new Volume<float>(argv[1]);
    float3 spacing = input->getSpacing();

    std::cout << "Dataset of size " << input->getWidth() << ", " << input->getHeight() << ", " << input->getDepth() << " loaded "<< std::endl;

    // Set initial mask
    int3 seedPosition(atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    float seedRadius = atof(argv[6]);

    // Do level set
    try {
        Volume<float> * res = runLevelSet(ocl, input, seedPosition, seedRadius, atoi(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10]));

        // Visualize result
        if(argc == 13) {
            float level = atof(argv[11]);
            float window = atof(argv[12]);
            visualize(input, res, level, window);
        }

        // Store result
        Volume<char> * segmentation = new Volume<char>(res->getSize());
        segmentation->setSpacing(spacing);
        for(int i = 0; i < res->getTotalSize(); i++) {
            if(res->get(i) < 0.0f) {
                segmentation->set(i, 1);
            } else {
                segmentation->set(i, 0);
            }
        }

        segmentation->save(argv[2]);

    } catch(cl::Error &e) {
        cout << "OpenCL error occurred: " << e.what() << " " << getCLErrorString(e.err()) << endl;
    }

}

#include "levelSet.hpp"
#include "histogram-pyramids.hpp"
#include <string>
#include <iostream>
#include "config.h"
using namespace SIPL;

void updateLevelSetFunction(
        OpenCL &ocl,
        cl::Kernel &kernel,
        cl::Image3D &input,
        cl::Buffer &positions,
        int activeVoxels,
        int numberOfThreads,
        int groupSize,
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
            cl::NDRange(numberOfThreads),
            cl::NDRange(groupSize)
    );
}


/*
void visualizeActiveSet(OpenCL &ocl, cl::Image3D &activeSet, int3 size) {
    char * data = new char[size.x*size.y*size.z];
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;


    ocl.queue.enqueueReadImage(
            activeSet,
            CL_TRUE,
            origin,
            region,
            0, 0,
            data
    );

    Volume<char> * activeSetImage = new Volume<char>(size);
    activeSetImage->setData(data);
    activeSetImage->display();
}

void visualizeSpeedFunction(OpenCL &ocl, cl::Image3D &speedFunction, int3 size) {
    float * data = new float[size.x*size.y*size.z];
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;


    ocl.queue.enqueueReadImage(
            speedFunction,
            CL_TRUE,
            origin,
            region,
            0, 0,
            data
    );

    Volume<float> * activeSetImage = new Volume<float>(size);
    activeSetImage->setData(data);
    activeSetImage->display();
}
*/


SIPL::Volume<char> * runLevelSet(
        const char * filename,
        int3 seedPos,
        float seedRadius,
        int iterations,
        float threshold,
        float epsilon,
        float alpha
        ) {

    // Create OpenCL context
    OpenCL ocl;
    ocl.context = createCLContext(CL_DEVICE_TYPE_GPU, VENDOR_ANY);
    VECTOR_CLASS<cl::Device> devices = ocl.context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    ocl.device = devices[0];
    ocl.queue = cl::CommandQueue(ocl.context, devices[0]);
    std::string kernelFilename = std::string(KERNELS_DIR) + std::string("/kernels.cl");
    std::string buildOptions = "";
    if(ocl.device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") == 0)
        buildOptions = "-DNO_3D_WRITE";
    ocl.program = buildProgramFromSource(ocl.context, kernelFilename);

    // Load volume
    Volume<float> * input = new Volume<float>(filename);
    float3 spacing = input->getSpacing();

    std::cout << "Dataset of size " << input->getWidth() << ", " << input->getHeight() << ", " << input->getDepth() << " loaded "<< std::endl;

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

    cl::Image3D borderSet = cl::Image3D(
            ocl.context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_R, CL_SIGNED_INT8),
            input->getWidth(),
            input->getHeight(),
            input->getDepth()
    );

    // Create seed
    char narrowBandDistance = 4;
    cl::Kernel createSeedKernel(ocl.program, "initializeLevelSetFunction");
    createSeedKernel.setArg(0, phi_1);
    createSeedKernel.setArg(1, seedPos.x);
    createSeedKernel.setArg(2, seedPos.y);
    createSeedKernel.setArg(3, seedPos.z);
    createSeedKernel.setArg(4, seedRadius);
    createSeedKernel.setArg(5, activeSet);
    createSeedKernel.setArg(6, narrowBandDistance);
    createSeedKernel.setArg(7, phi_2);
    createSeedKernel.setArg(8, borderSet);
    ocl.queue.enqueueNDRangeKernel(
            createSeedKernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
    );


    cl::Kernel init3DImage(ocl.program, "init3DImage");
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = size.x;
    region[1] = size.y;
    region[2] = size.z;
    cl::Kernel updateActiveSetKernel(ocl.program, "updateActiveSet");
    cl::Kernel kernel(ocl.program, "updateLevelSetFunction");
    cl::Kernel updateBorderSetKernel(ocl.program, "updateBorderSet");


    HistogramPyramid3D * hp = new HistogramPyramid3D(ocl);
    const int groupSize = 128;
    //const float timestep = 1.0f;
    //const int levelSetUpdates = 4*narrowBandDistance / timestep;
    //visualizeActiveSet(ocl, activeSet, size);
    int narrowBands = 1000;
    for(int i = 0; i < narrowBands; i++) {
        //if(i % 10 == 0)
        //visualizeActiveSet(ocl, activeSet, size);
        hp->create(activeSet, size.x, size.y, size.z);
        int activeVoxels = hp->getSum();
        if(activeVoxels == 0)
            break;
        int numberOfThreads = activeVoxels+groupSize-(activeVoxels-(activeVoxels / groupSize)*groupSize);
        std::cout << "Number of active voxels: " << activeVoxels << std::endl;
        cl::Buffer positions = hp->createPositionBuffer();

        for(int j = 0; j < iterations; j++) {
            if(j % 2 == 0) {
                updateLevelSetFunction(ocl, kernel, inputData, positions, activeVoxels, numberOfThreads, groupSize, phi_1, phi_2, threshold, epsilon, alpha);
            } else {
                updateLevelSetFunction(ocl, kernel, inputData, positions, activeVoxels, numberOfThreads, groupSize, phi_2, phi_1, threshold, epsilon, alpha);
            }
        }

        cl::Image3D activeSet2 = cl::Image3D(
                ocl.context,
                CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_R, CL_SIGNED_INT8),
                input->getWidth(),
                input->getHeight(),
                input->getDepth()
        );

        init3DImage.setArg(0, activeSet2);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
        );

        // Create new active set
        updateActiveSetKernel.setArg(0, positions);
        updateActiveSetKernel.setArg(1, phi_1);
        updateActiveSetKernel.setArg(2, activeSet2);
        updateActiveSetKernel.setArg(3, narrowBandDistance);
        updateActiveSetKernel.setArg(4, activeSet);
        updateActiveSetKernel.setArg(5, borderSet);
        updateActiveSetKernel.setArg(6, activeVoxels);
        ocl.queue.enqueueNDRangeKernel(
            updateActiveSetKernel,
            cl::NullRange,
            cl::NDRange(numberOfThreads),
            cl::NDRange(groupSize)
        );

        activeSet = activeSet2;

        // Update border set
        init3DImage.setArg(0, borderSet);
        ocl.queue.enqueueNDRangeKernel(
            init3DImage,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
        );

        updateBorderSetKernel.setArg(0, borderSet);
        updateBorderSetKernel.setArg(1, phi_1);
        ocl.queue.enqueueNDRangeKernel(
            updateBorderSetKernel,
            cl::NullRange,
            cl::NDRange(size.x,size.y,size.z),
            cl::NullRange
        );

        hp->deleteHPlevels();
    }
    std::cout << "Finished level set iterations" << std::endl;


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

    Volume<char> * segmentation = new Volume<char>(phi->getSize());
    segmentation->setSpacing(spacing);
    for(int i = 0; i < phi->getTotalSize(); i++) {
        if(phi->get(i) < 0.0f) {
            segmentation->set(i, 1);
        } else {
            segmentation->set(i, 0);
        }
    }
    std::cout << "Finished transfering data back to host." << std::endl;


    return segmentation;
}


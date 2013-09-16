#include "levelSet.hpp"
#include <iostream>
using namespace std;

int main(int argc, char ** argv) {

    if(argc < 10) {
        cout << endl;
        cout << "OpenCL Level Set Segmentation by Erik Smistad 2013" << endl;
        cout << "www.github.com/smistad/OpenCL-Level-Set-Segmentation/" << endl;
        cout << "======================================================" << endl;
        cout << "The speed function is defined as -alpha*(epsilon-(T-intensity))+(1-alpha)*curvature" << endl;
        cout << "Usage: " << argv[0] << " inputFile.mhd seedX seedY seedZ seedRadius iterations threshold epsilon alpha [level window] [outputFile.mhd]" << endl;
        cout << "If the level and window arguments are set, the segmentation result will be displayed as an overlay to the input volume " << endl;
        return -1;
    }

    // Set initial mask
    SIPL::int3 seedPosition(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    float seedRadius = atof(argv[5]);
    float level = -1.0f;
    float window = -1.0f;
    if(argc == 14 || argc == 12) {
        level = atof(argc == 12 ? argv[10] : argv[11]);
        window = atof(argc == 12 ? argv[11] : argv[12]);
    }
    std::string outputFilename = "";
    if(argc == 13 || argc == 11) { // filename specified, write to disk
        outputFilename = argc == 11 ? argv[10] : argv[13];
    }

    // Do level set
    try {
        runLevelSet(
                argv[1],
                seedPosition,
                seedRadius,
                atoi(argv[6]),
                atof(argv[7]),
                atof(argv[8]),
                atof(argv[9]),
                window != -1, // visualize
                level,
                window,
                outputFilename
        );
    } catch(cl::Error &e) {
        cout << "OpenCL error occurred: " << e.what() << " " << getCLErrorString(e.err()) << endl;
    }

}

#ifndef LEVELSET_HPP_
#define LEVELSET_HPP_

#include "SIPL/Core.hpp"
#include "commons.hpp"

void runLevelSet(
        const char * filename,
        SIPL::int3 seedPos,
        float seedRadius,
        int iterations,
        float threshold,
        float epsilon,
        float alpha,
        bool visualizeResult,
        float level,
        float window,
        std::string outputFilename
);



#endif /* LEVELSET_HPP_ */

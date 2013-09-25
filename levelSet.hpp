#ifndef LEVELSET_HPP_
#define LEVELSET_HPP_

#include "SIPL/Core.hpp"
#include "commons.hpp"

SIPL::Volume<char> * runLevelSet(
        const char * filename,
        SIPL::int3 seedPos,
        float seedRadius,
        int iterations,
        float threshold,
        float epsilon,
        float alpha
);



#endif /* LEVELSET_HPP_ */

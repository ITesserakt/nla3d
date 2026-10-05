// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#include "materials/MaterialFactory.h"

namespace nla3d {

const char* const MaterialFactory::matModelLabels[] = {"UNDEFINED", "Neo-Hookean", "Biderman", "Mooney-Rivlin"};

MaterialFactory::matId MaterialFactory::matName2matId(const std::string& matName) {
    for (uint16 i = 0; i < LAST; i++) {
        if (matName == matModelLabels[i]) {
            return static_cast<matId>(i);
        }
    }
    return NOT_DEFINED;
}

Material* MaterialFactory::createMaterial(const std::string& matName) {
    const uint16 matId = matName2matId(matName);
    Material* mat;
    if (matId == NOT_DEFINED) {
        LOG(ERROR) << "Can't find material " << matName;
        return nullptr;
    }
    switch (matId) {
    case NEO_HOOKEAN_COMP:
        mat = new Mat_Comp_Neo_Hookean();
        break;
    case BIDERMAN_COMP:
        mat = new Mat_Comp_Biderman();
        break;
    case MOONEYRIVLIN_COMP:
        mat = new Mat_Comp_MooneyRivlin();
        break;
    default:
        LOG(ERROR) << "Don't have a material with id  = " << matId;
        return nullptr;
    }
    mat->code = matId;
    return mat;
}

} // namespace nla3d

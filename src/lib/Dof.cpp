#include "Dof.h"

namespace nla3d {

const uint16 Dof::numberOfDofTypes = UNDEFINED;
const char* const Dof::dofTypeLabels[] = {"UX",   "UY",       "UZ", "ROTX", "ROTY", "ROTZ", "HYDRO_PRESSURE",
                                          "TEMP", "UNDEFINED"};

// Check that TypeLabels has the same length as numberOfDofTypes
static_assert(Dof::numberOfDofTypes == sizeof(Dof::dofTypeLabels) / sizeof(Dof::dofTypeLabels[0]) - 1,
              "dofTypeLabels and dofType must have the same number of elements");

Dof::dofType Dof::label2dofType(const std::string& label) {
    for (uint16 i = 0; i < numberOfDofTypes; i++) {
        if (label == dofTypeLabels[i]) {
            return static_cast<dofType>(i);
        }
    }
    LOG(ERROR) << "Unknown dof key = " << label;
    return UNDEFINED;
}

void DofCollection::initDofTable(const uint32 _numberOfEntities) {
    clearDofTable();
    numberOfEntities = _numberOfEntities;
    dofPos.assign(numberOfEntities + 1, 0);
}

// n index starts from 1
void DofCollection::addDof(const uint32 n, const std::initializer_list<Dof::dofType> _dofs) {
    assert(n <= numberOfEntities);
    assert(!dofPos.empty());
    std::set<Dof::dofType> dofs(_dofs);
    std::vector<Dof> newDofs;
    // if already registered - just return
    if (dofPos[n - 1] == dofPos[n]) {
        for (auto& v : dofs)
            newDofs.emplace_back(v);
    } else {
        for (auto& v : dofs) {
            bool isFound = false;
            for (auto it = this->dofs.begin() + dofPos[n - 1]; it < this->dofs.begin() + dofPos[n]; ++it) {
                if (it->type == v) {
                    isFound = true;
                    break;
                }
            }
            if (!isFound) {
                newDofs.emplace_back(v);
            }
        }
    }

    if (newDofs.empty())
        return;

    this->dofs.insert(this->dofs.begin() + dofPos[n], newDofs.begin(), newDofs.end());
    uniqueDofTypes.insert(std::begin(dofs), std::end(dofs));

    // incement dofPos with newDofs.size() for all above
    uint32 i = n;
    while (i <= numberOfEntities)
        dofPos[i++] += newDofs.size();

    numberOfUsedDofs += newDofs.size();
}

void DofCollection::clearDofTable() {
    dofs.clear();
    dofPos.clear();
    uniqueDofTypes.clear();
    numberOfUsedDofs = 0;
    numberOfEntities = 0;
}

// n index starts from 1
bool DofCollection::isDofUsed(const uint32 n, const Dof::dofType dof) {
    assert(n <= numberOfEntities);
    assert(!dofPos.empty());
    if (dofPos[n - 1] == dofPos[n])
        return false;
    for (auto it = dofs.begin() + dofPos[n - 1]; it < dofs.begin() + dofPos[n]; ++it) {
        if (it->type == dof)
            return true;
    }
    return false;
}

// return nullptr if Dof was not found
Dof* DofCollection::getDof(const uint32 n, const Dof::dofType dof) {
    assert(n <= numberOfEntities);
    assert(!dofPos.empty());

    Dof* pdof = nullptr;

    for (auto it = dofs.begin() + dofPos[n - 1]; it < dofs.begin() + dofPos[n]; it++) {
        if (it->type == dof) {
            pdof = &(*it);
            break;
        }
    }
    return pdof;
}

std::set<Dof::dofType> DofCollection::getUniqueDofTypes() { return uniqueDofTypes; }

} // namespace nla3d

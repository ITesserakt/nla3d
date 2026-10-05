// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#include "Node.h"

namespace nla3d {

void Node::display(const uint32 nn) const { LOG(INFO) << "N " << nn << " : " << pos; }

std::string Node::toString() {
    std::stringstream ss;
    ss << pos[0] << ' ' << pos[1] << ' ' << pos[2];
    return ss.str();
}

} // namespace nla3d

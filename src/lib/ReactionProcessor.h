// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once
#include "FEStorage.h"
#include "PostProcessor.h"
#include "sys.h"

namespace nla3d {

class ReactionProcessor : public PostProcessor {
  public:
    explicit ReactionProcessor(FEStorage& st);
    ReactionProcessor(FEStorage& st, std::string _filename);
    ~ReactionProcessor() override = default;

    void pre() override;
    void process(uint16 curLoadstep) override;
    void post(uint16 curLoadstep) override;

    std::vector<double> getReactions(Dof::dofType dof);

    std::vector<uint32> nodes;
    std::vector<Dof::dofType> dofs;

  protected:
    std::string filename;
    std::vector<std::vector<double>> sumOfDofsReactions;
};

} // namespace nla3d

use platform::collections::hash_set::HashSet;
use platform::collections::vec::Vec;
use strum::{Display, EnumString};

#[derive(EnumString, Display, Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[strum(serialize_all = "UPPERCASE")]
pub enum DofKind {
    UX,
    UY,
    UZ,
    RotX,
    RotY,
    RotZ,
}

#[derive(Debug)]
pub struct Dof {
    kind: DofKind,
    is_constrained: bool,
    equation_num: u32,
}

#[derive(Debug)]
pub struct DofCollection {
    unique_dof_kinds: HashSet<DofKind>,
    dofs: Vec<Dof>,
    dof_positions: Vec<u32>,
}

impl DofCollection {
    pub fn add_dof(&mut self, n: u32, kinds: impl IntoIterator<Item = DofKind>) {
        todo!()
    }

    pub fn clear(&mut self) {
        todo!()
    }

    pub fn get_dof(&self, n: u32, kind: DofKind) -> Option<&Dof> {
        todo!()
    }
}

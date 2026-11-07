#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;
#[cfg(feature = "std")]
extern crate std;

pub mod compat {
    #[cfg(feature = "std")]
    pub use std::{borrow, boxed, string, sync};

    #[cfg(not(feature = "std"))]
    pub use alloc::{borrow, boxed, string, sync};
}

pub mod collections {
    #[cfg(feature = "std")]
    pub use std::{collections::*, vec};

    #[cfg(not(feature = "std"))]
    pub use alloc::{collections::*, vec};

    pub mod hash_set {
        pub use hashbrown::hash_set::*;
        pub type HashSet<T, S = foldhash::fast::RandomState> = hashbrown::hash_set::HashSet<T, S>;
    }

    pub mod hash_map {
        pub use hashbrown::hash_map::*;
        pub type HashMap<K, V, S = foldhash::fast::RandomState> =
            hashbrown::hash_map::HashMap<K, V, S>;
    }
}

#[cfg(all(not(feature = "std"), not(feature = "mimalloc")))]
#[global_allocator]
static A: libc_alloc::LibcAlloc = libc_alloc::LibcAlloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static A: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    log::error!("{info}");
    unsafe { libc::abort() }
}

use compat::boxed::Box;
use collections::vec::Vec;

type DynLog = Box<dyn log::Log>;

#[derive(Default)]
struct LogSinks(Vec<DynLog>);

pub struct ConfigureScope<'a>(&'a mut Vec<DynLog>);

impl ConfigureScope<'_> {
    pub fn add_logger(&mut self, logger: impl log::Log + 'static) {
        self.0.push(Box::new(logger));
    }
}

impl log::Log for LogSinks {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.0.iter().any(|it| it.enabled(metadata))
    }

    fn log(&self, record: &log::Record) {
        for logger in &self.0 {
            logger.log(record);
        }
    }

    fn flush(&self) {
        for logger in &self.0 {
            logger.flush();
        }
    }
}

pub fn initialize_logger(
    scope: impl for<'scope> FnOnce(ConfigureScope<'scope>),
) -> Result<(), log::SetLoggerError> {
    let mut logger = LogSinks::default();
    scope(ConfigureScope(&mut logger.0));
    let static_logger = Box::leak(Box::new(logger));

    log::set_logger(static_logger)
}

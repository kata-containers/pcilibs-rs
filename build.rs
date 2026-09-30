#[cfg(feature = "std")]
#[path = "build_support.rs"]
mod linux;

fn main() {
    #[cfg(feature = "std")]
    linux::generate();
}

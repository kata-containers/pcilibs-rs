// SPDX-License-Identifier: Apache-2.0

fn main() -> std::io::Result<()> {
    let detected = pcilibs_rs::platform::discover(&pcilibs_rs::Sysfs::default())?;
    println!("{:?}", detected.platform);
    Ok(())
}

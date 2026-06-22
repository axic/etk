use etk_cli::errors::WithSources;
use etk_cli::io::HexWrite;

use etk_asm::ingest::Ingest;

use std::fs::{self, File};
use std::io::prelude::*;
use std::path::{Path, PathBuf};

use clap::StructOpt;

/// Required height of any GIF passed to `--evmgif`. Matches the reference
/// `inject.py` from the evm.gif repository.
const REQUIRED_HEIGHT: u16 = 867;

/// `21 FF 06 'EVMGIF'` — the GIF Application Extension introducer the
/// assembler is required to emit at the start of every `--evmgif` payload.
/// We re-check this defensively before splicing the payload into the host
/// GIF so a future change in the assembler's output framing can't silently
/// produce a malformed file.
const APP_EXT_HEADER: [u8; 9] = [
    0x21, 0xFF, 0x06, 0x45, 0x56, 0x4D, 0x47, 0x49, 0x46,
];

#[derive(Debug, StructOpt)]
#[structopt(name = "eas")]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,
    #[structopt(parse(from_os_str))]
    out: Option<PathBuf>,

    /// Inject the assembled bytecode into a GIF89a host file.
    ///
    /// Opens the given GIF, validates that the height is 867 and that a
    /// Global Color Table is present, then assembles `<input>` in
    /// `--evmgif` chunked form anchored at the byte position where the
    /// EVMGIF Application Extension will be inserted (`len(gif) - 1`,
    /// in place of the trailing `0x3B` GIF trailer). Palette entries
    /// 0 and 1 of the host GIF are overwritten with
    /// `POP POP PUSH2 <jump_offset> JUMP`, where `<jump_offset>` lands
    /// on the preamble JUMPDEST inside the inserted Application
    /// Extension. The Application Extension is then spliced in
    /// immediately before the trailer, and the resulting combined GIF
    /// is written to `<out>` (or stdout if omitted) as raw binary
    /// instead of hex.
    #[structopt(long = "evmgif", value_name = "GIF", parse(from_os_str))]
    evmgif: Option<PathBuf>,
}

fn create(path: &Path) -> File {
    match File::create(path) {
        Err(why) => panic!("couldn't create `{}`: {}", path.display(), why),
        Ok(file) => file,
    }
}

fn main() {
    if let Err(msg) = run() {
        eprintln!("{}", msg);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let opt: Opt = clap::Parser::parse();

    match opt.evmgif.as_deref() {
        Some(gif_path) => run_evmgif(gif_path, &opt.input, opt.out.as_deref()),
        None => run_plain(&opt.input, opt.out.as_deref()),
    }
}

fn run_plain(input: &Path, out: Option<&Path>) -> Result<(), String> {
    let mut output: Box<dyn Write> = match out {
        Some(o) => Box::new(create(o)),
        None => Box::new(std::io::stdout()),
    };

    let hex_out = HexWrite::new(&mut output);

    let mut ingest = Ingest::new(hex_out);
    ingest
        .ingest_file(input.to_path_buf())
        .map_err(|e| format!("{}", WithSources(e)))?;

    output.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}

fn run_evmgif(gif_path: &Path, etk_path: &Path, out: Option<&Path>) -> Result<(), String> {
    let src = fs::read(gif_path)
        .map_err(|e| format!("reading GIF `{}`: {}", gif_path.display(), e))?;

    if src.len() < 14 || &src[..6] != b"GIF89a" {
        return Err("not a GIF89a file".into());
    }

    let height = u16::from_le_bytes([src[8], src[9]]);
    if height != REQUIRED_HEIGHT {
        return Err(format!(
            "height must be {}, got {}",
            REQUIRED_HEIGHT, height
        ));
    }

    if src[10] & 0x80 == 0 {
        return Err("input has no Global Color Table; palette entry 0 missing".into());
    }

    if *src.last().unwrap() != 0x3B {
        return Err("input does not end with GIF trailer 0x3B".into());
    }

    // The Application Extension is spliced in immediately before the
    // trailing 0x3B, so its leading magic byte lands at the index
    // currently occupied by the trailer.
    let app_ext_offset = src.len() - 1;
    // The preamble JUMPDEST sits 10 bytes into the Application Extension:
    // 9 bytes of magic (`21 FF 06 'EVMGIF'`) + the 1-byte `<size_0>` byte.
    let jump_offset = app_ext_offset + 10;
    if jump_offset > 0xFFFF {
        return Err(format!(
            "jump_offset {} does not fit in 16 bits",
            jump_offset
        ));
    }

    let mut app_ext: Vec<u8> = Vec::new();
    {
        let mut ingest = Ingest::with_evmgif_offset(&mut app_ext, app_ext_offset as u64);
        ingest
            .ingest_file(etk_path.to_path_buf())
            .map_err(|e| format!("{}", WithSources(e)))?;
    }

    if !app_ext.starts_with(&APP_EXT_HEADER) {
        return Err(format!(
            "assembler output does not start with APP_EXT_HEADER ({})",
            hex::encode(APP_EXT_HEADER)
        ));
    }

    let mut combined = src;
    // POP, POP — clean up the two bytes the GIF header pushes (logical
    // screen dimensions) before our jump.
    combined[13] = 0x50;
    combined[14] = 0x50;
    // PUSH2 <jump_offset>
    combined[15] = 0x61;
    combined[16] = ((jump_offset >> 8) & 0xFF) as u8;
    combined[17] = (jump_offset & 0xFF) as u8;
    // JUMP
    combined[18] = 0x56;

    combined.splice(app_ext_offset..app_ext_offset, app_ext);

    match out {
        Some(p) => fs::write(p, &combined)
            .map_err(|e| format!("writing output `{}`: {}", p.display(), e))?,
        None => std::io::stdout()
            .write_all(&combined)
            .map_err(|e| format!("writing to stdout: {}", e))?,
    }

    Ok(())
}

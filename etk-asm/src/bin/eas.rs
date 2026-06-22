use etk_cli::errors::WithSources;
use etk_cli::io::HexWrite;

use etk_asm::ingest::{Error, Ingest};

use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

use clap::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "eas")]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,
    #[structopt(parse(from_os_str))]
    out: Option<PathBuf>,

    /// Emit `--evmgif` chunked output at the given base byte offset.
    ///
    /// Prepends the 9-byte magic header `21 FF 06 'EVMGIF'`, then the
    /// 2-byte preamble `<size> JUMPDEST`, then the user's bytecode
    /// interleaved with `PUSH1 <size> POP` headers so each inter-header
    /// span is at most 256 bytes. `OFFSET` is the byte position of the
    /// leading `0x21` magic byte in the wrapping file; the caller jumps
    /// to the preamble JUMPDEST at `OFFSET + 10` and the user's first
    /// instruction sits at `OFFSET + 11`. The caller is responsible for
    /// cleaning up any stack effect from the `<size_0>` byte before
    /// jumping in. Every label-bearing push resolves to its chunked
    /// position plus that base offset. Accepts decimal or `0x`-prefixed
    /// hexadecimal.
    #[structopt(long = "evmgif", value_name = "OFFSET", parse(try_from_str = parse_offset))]
    evmgif: Option<u64>,
}

fn parse_offset(src: &str) -> Result<u64, std::num::ParseIntError> {
    match src.strip_prefix("0x").or_else(|| src.strip_prefix("0X")) {
        Some(rest) => u64::from_str_radix(rest, 16),
        None => src.parse::<u64>(),
    }
}

fn create(path: PathBuf) -> File {
    match File::create(&path) {
        Err(why) => panic!("couldn't create `{}`: {}", path.display(), why),
        Ok(file) => file,
    }
}

fn main() {
    let err = match run() {
        Ok(_) => return,
        Err(e) => e,
    };

    eprintln!("{}", WithSources(err));
    std::process::exit(1);
}

fn run() -> Result<(), Error> {
    let opt: Opt = clap::Parser::parse();

    let mut out: Box<dyn Write> = match opt.out {
        Some(o) => Box::new(create(o)),
        None => Box::new(std::io::stdout()),
    };

    let hex_out = HexWrite::new(&mut out);

    let mut ingest = match opt.evmgif {
        Some(offset) => Ingest::with_evmgif_offset(hex_out, offset),
        None => Ingest::new(hex_out),
    };
    ingest.ingest_file(opt.input)?;

    out.write_all(b"\n").unwrap();

    Ok(())
}
